import warnings

import numpy as np
import scipy.stats
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.statistics import Statistics


def extract_params_from_journal_Lorenz95(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['theta1']).reshape(-1, 1), np.array(params['theta2']).reshape(-1, 1),
                           np.array(params['sigma_e']).reshape(-1, 1), np.array(params['phi']).reshape(-1, 1)), axis=1)


def extract_params_and_weights_from_journal_Lorenz95(jrnl, step=None):
    params = jrnl.get_parameters(step)
    weights = jrnl.get_weights(step)
    return np.concatenate((np.array(params['theta1']).reshape(-1, 1), np.array(params['theta2']).reshape(-1, 1),
                           np.array(params['sigma_e']).reshape(-1, 1), np.array(params['phi']).reshape(-1, 1)),
                          axis=1), weights.reshape(-1)


def extract_posterior_mean_from_journal_Lorenz95(jrnl, step=None):
    post_mean = jrnl.posterior_mean(step)
    return np.array((post_mean['theta1'], post_mean['theta2'], post_mean['sigma_e'], post_mean['phi']))


class LorenzLargerStatistics(Statistics):

    def __init__(self, moments=3, auto_covariances=5, covariances=5, cross_covariances=5, degree=1, cross=False):
        """We compute here moments, covariances, cross covariances, autocovariances."""
        self.number_moments = moments
        self.number_auto_covariances = auto_covariances
        self.number_covariances = covariances
        self.number_cross_covariances = cross_covariances
        self.degree = degree
        self.cross = cross

    def statistics(self, data):
        if isinstance(data, list):
            if np.array(data).shape == (len(data),):
                if len(data) == 1:
                    data = np.array(data).reshape(1, 1)
                data = np.array(data).reshape(len(data), 1)
            else:
                data = np.concatenate(data).reshape(len(data), -1)
        else:
            raise TypeError('Input data should be of type list, but found type {}'.format(type(data)))
        # Extract Summary Statistics; this here assumes explicitly we are using 40 variables.
        num_element, timestep = len(data), int(data[0].shape[0] / 40)
        result = np.zeros(shape=(num_element, self.number_moments + self.number_auto_covariances +
                                 self.number_covariances + 2 * self.number_cross_covariances))
        # Compute statistics
        if np.any(np.isinf(data)):
            warnings.warn("Infinity in the output of Lorenz95 model", RuntimeWarning)
        if np.any(np.isnan(data)):
            warnings.warn("nan in the output of Lorenz95 model", RuntimeWarning)

        for ind_element in range(0, num_element):
            # First convert the vector to the 40 dimensional timeseries
            data_ind_element = data[ind_element].reshape(40, timestep)
            # compute 10 central moments:
            moments = np.zeros(self.number_moments)
            moments[0] = np.mean(np.mean(data_ind_element, 1))
            for i in range(1, self.number_moments):
                moments[i] = np.mean(scipy.stats.moment(data_ind_element, axis=1, moment=i + 1))
                # second centered moment corresponds to variance

            # auto-covariances: covariance between one timeseries and itself with some lag
            # Here we repeat computation, as we could return the auto-covariances for all lags with
            # one single function call. It does not matter too much for now.
            auto_covariances = np.zeros(self.number_auto_covariances)
            for ind in range(0, data_ind_element.shape[0]):  # loop over the different components of timeseries.
                for i in range(self.number_auto_covariances):
                    auto_covariances[i] += self._covariance_lag(data_ind_element[ind, :],
                                                                data_ind_element[ind, :], lag=i + 1)
            auto_covariances /= data_ind_element.shape[0]

            # covariances between different timeseries with lag 0 and spatial distance up to 5, with cyclic
            # boundary conditions.
            covariances_neighbors = np.zeros(self.number_covariances)
            for ind in range(0, data_ind_element.shape[0]):  # loop over the different components of timeseries.
                for i in range(self.number_covariances):
                    # the % is for the cyclic conditions
                    covariances_neighbors[i] += self._covariance_lag(
                        data_ind_element[ind, :], data_ind_element[(ind + i + 1) % data_ind_element.shape[0], :], lag=0)
            covariances_neighbors /= data_ind_element.shape[0]

            # we now extract cross-covariances with both time lag and spatial distance: consider timeseries at distance
            # i and use lag i, by keeping them to be the same:
            cross_covariances_neighbors_right = np.zeros(self.number_cross_covariances)
            cross_covariances_neighbors_left = np.zeros(self.number_cross_covariances)
            for ind in range(0, data_ind_element.shape[0]):  # loop over the different components of timeseries.
                for i in range(self.number_cross_covariances):
                    # the % is for the cyclic conditions
                    cross_covariances_neighbors_right[i] += self._covariance_lag(
                        data_ind_element[ind, :], data_ind_element[(ind + i + 1) % data_ind_element.shape[0], :],
                        lag=i + 1)
                    cross_covariances_neighbors_left[i] += self._covariance_lag(
                        data_ind_element[(ind + i + 1) % data_ind_element.shape[0], :], data_ind_element[ind, :],
                        lag=i + 1)
            cross_covariances_neighbors_right /= data_ind_element.shape[0]
            cross_covariances_neighbors_left /= data_ind_element.shape[0]

            result[ind_element] = np.concatenate((moments, auto_covariances, covariances_neighbors,
                                                  cross_covariances_neighbors_right, cross_covariances_neighbors_left))

            for i, stat in enumerate(result[ind_element]):
                if np.any(np.isinf(stat)):
                    warnings.warn("Infinity in stat number {}".format(i + 1), RuntimeWarning)
                if np.any(np.isnan(stat)):
                    warnings.warn("nan in stat number {}".format(i + 1), RuntimeWarning)
            # Expand the data with polynomial expansion
        result = self._polynomial_expansion(result)
        return np.array(result)

    def _covariance_lag(self, x, y, lag=0):
        """ Computes cross-covariance between x and y. Using np.correlate is faster than using the for loop over
        elements. This function can be used to compute all kinds of covariances/autocovariancess/cross-covariances
        needed.

        Parameters
        ----------
        x: numpy.ndarray
            Vector of real numbers.
        y: numpy.ndarray
            Vector of real numbers.

        Returns
        -------
        numpy.ndarray
            Cross-covariance calculated between x and y.
        """
        assert x.shape[0] == y.shape[0]  # we consider case in which the timeseries all have same length
        # data_size = np.int(x.shape[0] / 2)  # it may work only for timeseries with even number of timesteps.
        # the proper definition of covariance also subtracts the means of the two series that you are considering.
        if lag == 0:
            return np.correlate(x, y, mode="valid")[0] / x.shape[0] - np.mean(x) * np.mean(y)
        else:
            return np.correlate(x[lag:], y[:-lag], mode="valid")[0] / (x.shape[0] - lag) - np.mean(x[lag:]) * np.mean(
                y[:-lag])
            # this in case you want to return different lags at once; don't care about that for now.
            # return np.correlate(x, y, mode="same")[data_size + 1: data_size + self.order + 1]  # many correlations at once
            # return np.correlate(x, y, mode="same")[data_size + lag] / (x.shape[0] - lag) - np.mean(x[lag:]) * np.mean(
            #     y[:-lag] if lag > 0 else y)  # x[:-0] does not work


class StochLorenz95(ProbabilisticModel, Continuous):
    """Generates time dependent 'slow' weather variables following forecast model of Wilks [1],
        a stochastic reparametrization of original Lorenz model Lorenz [2].

        [1] Wilks, D. S. (2005). Effects of stochastic parametrizations in the lorenz ’96 system.
        Quarterly Journal of the Royal Meteorological Society, 131(606), 389–407.

        [2] Lorenz, E. (1995). Predictability: a problem partly solved. In Proceedings of the
        Seminar on Predictability, volume 1, pages 1–18. European Center on Medium Range
        Weather Forecasting, Europe

        Parameters
        ----------
        parameters: list
            Contains the closure parameters of the parametrization: [theta_1, theta_2, sigma_e, phi].
        initial_state: numpy.ndarray, optional
            Initial state value of the time-series, The default value is None, which assumes a previously computed
            value from a full Lorenz model as the Initial value.
        F, b, h, c: floats, optional
            These represent some constants used in the definition of the ODEs.
        K: integer, optional
            The total number of slow variables
        J: integer, optional
            The number of fast variables for each slow variable
        time_units: float, optional
            Number of time unites for which to run the system; default to 4, whih corresponds roughly to 20 days.
        n_timestep_per_time_unit: int, optional
            Number of steps for a time unit. The default value is 30 steps.
        name: string, optional
            Name of the model
        """

    def __init__(self, parameters, initial_state=None, F=10, b=10, h=1, c=4, J=8, K=40, time_units=4,
                 n_timestep_per_time_unit=30, name='StochLorenz95'):
        # settings in Arnold et al. K=8, J=32, h=1, F=20, b=10, c=10 or c=4
        self._set_initial_state(initial_state)
        self.F = F  # sometimes this parameter is given value 20; it is given value 10 in Hakkarainen et al. (2012);
        # either 18 or 20 in Wilks (2005); 20 in Arnold et al. (2013)
        # self.sigma_e = 1
        # self.phi = 0.4

        # define the parameters of the true model:
        self.b = b
        self.h = h
        self.c = c  # 10 is also used sometimes; it corresponds to the easy case, while c=4 is harder.
        self.J = J
        self.K = K
        self.time_units = time_units
        self.n_timestep_per_time_unit = n_timestep_per_time_unit
        self.n_timestep = int(self.time_units * self.n_timestep_per_time_unit)
        self.hc_over_b = self.h * self.c / self.b
        self.cb = self.c * self.b

        if not isinstance(parameters, list):
            raise TypeError('Input of StochLorenz95 model is of type list')

        if len(parameters) != 4:
            raise RuntimeError('Input list must be of length 4, containing [theta1, theta2, sigma_e, phi].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _set_initial_state(self, initial_state, lam=0):
        """This sets the initial state to a fixed value (if that was not provided). lam is the magnitude of the gaussian
        noise that is added to the initial state; it is used only in the Ensemble model."""
        if initial_state is None:
            # Assign initial state; this has 40 variables, it is suitable for K=40. In case K<40 is used, the initial
            # state is considered to be self.initial_state[0:K]
            self.initial_state = np.array([6.4558, 1.1054, -1.4502, -0.1985, 1.1905, 2.3887, 5.6689, 6.7284, 0.9301,
                                           4.4170, 4.0959, 2.6830, 4.7102, 2.5614, -2.9621, 2.1459, 3.5761, 8.1188,
                                           3.7343, 3.2147, 6.3542, 4.5297, -0.4911, 2.0779, 5.4642, 1.7152, -1.2533,
                                           4.6262, 8.5042, 0.7487, -1.3709, -0.0520, 1.3196, 10.0623, -2.4885, -2.1007,
                                           3.0754, 3.4831, 3.5744, 6.5790])
        else:
            self.initial_state = initial_state
        if lam > 0:
            self.initial_state += np.random.normal(scale=lam, size=self.initial_state.shape)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 4:
            raise ValueError('Number of parameters of StochLorenz95 model must be 4.')

        # Check whether input is from correct domain
        theta1 = input_values[0]
        theta2 = input_values[1]
        sigma_e = input_values[2]
        phi = input_values[3]

        # if theta1 <= 0 or theta2 <= 0:
        # why? this does not make much sense, the parameters of the deterministic part could be smaller than 0
        #    return False

        if sigma_e < 0 or phi < 0 or phi > 1:
            return False

        return True

    def _check_output(self, values):
        # if not isinstance(values[0], np.ndarray):
        #     raise ValueError('Output of the normal distribution is always a number.')
        return True

    def get_output_dimension(self):
        return self.K * self.n_timestep

    def get_number_parameters(self):
        return 4

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        theta1 = input_values[0]
        theta2 = input_values[1]
        sigma_e = input_values[2]
        phi = input_values[3]

        # Do the actual forward simulation
        vector_of_k_samples = self.Lorenz95(theta1, theta2, sigma_e, phi, k, rng=rng)
        # Format the output to obey API
        result = [np.array([x]) for x in vector_of_k_samples]
        return result

    def forward_simulate_true_model(self, k, n_timestep_per_time_unit=None, rng=np.random.RandomState()):
        # Do the actual forward simulation
        vector_of_k_samples_x, vector_of_k_samples_y = self.Lorenz95True(k,
                                                                         n_timestep_per_time_unit=n_timestep_per_time_unit,
                                                                         rng=rng)
        # Format the output to obey API
        result_x = [np.array([x]) for x in vector_of_k_samples_x]
        result_y = [np.array([y]) for y in vector_of_k_samples_y]
        return result_x, result_y

    def Lorenz95(self, theta1, theta2, sigma_e, phi, k, rng=np.random.RandomState()):

        # Generate n_simulate time-series of weather variables satisfying Lorenz 95 equations
        result = []

        # Initialize timesteps.
        time_steps = np.linspace(0, self.time_units, self.n_timestep)

        for k in range(0, k):
            # Define a parameter object containing parameters which is needed
            # to evaluate the ODEs
            # Stochastic forcing term
            eta = sigma_e * np.sqrt(1 - pow(phi, 2)) * rng.normal(0, 1, self.initial_state.shape[0])

            # Initialize the time-series
            timeseries = np.zeros(shape=(self.K, self.n_timestep), dtype=np.float)
            timeseries[:, 0] = self.initial_state[0:self.K]
            # Compute the timeseries for each time steps
            for ind in range(0, self.n_timestep - 1):
                # parameters to be supplied to the ODE solver
                parameter = [eta, np.array([theta1, theta2])]
                # Each timestep is computed by using a 4th order Runge-Kutta solver
                x = self._rk4ode(self._l95ode_par, np.array([time_steps[ind], time_steps[ind + 1]]), timeseries[:, ind],
                                 parameter)
                timeseries[:, ind + 1] = x[:, -1]
                # Update stochastic forcing term
                eta = phi * eta + sigma_e * np.sqrt(1 - pow(phi, 2)) * rng.normal(0, 1)
            result.append(timeseries.flatten())
        # return an array of objects of type Timeseries
        return result

    def Lorenz95True(self, k, n_timestep_per_time_unit=None, rng=np.random.RandomState()):
        """"Note that here there is randomness in the choice of the starting value of the y variables. I chose them to
        be uniform in [0,1] at the beginning. """

        if n_timestep_per_time_unit is None:
            n_timestep = self.n_timestep
        else:
            n_timestep = int(self.time_units * n_timestep_per_time_unit)
        # Generate n_simulate time-series of weather variables satisfying Lorenz 95 equations
        result_X = []
        result_Y = []

        # Initialize timesteps
        # it is better to use a smaller timestep for the true model, as the solver may diverge otherwise.
        time_steps = np.linspace(0, self.time_units, n_timestep)

        # define the initial state of the Y variables. We take self.J fast variables per slow variable
        self.initial_state_Y = rng.uniform(size=(self.K * self.J))

        for k in range(0, k):
            # Define a parameter object containing parameters which is needed
            # to evaluate the ODEs

            # Initialize the time-series
            timeseries_X = np.zeros(shape=(self.K, n_timestep), dtype=np.float)
            timeseries_X[:, 0] = self.initial_state[0:self.K]

            timeseries_Y = np.zeros(shape=(self.initial_state_Y.shape[0], n_timestep), dtype=np.float)
            timeseries_Y[:, 0] = self.initial_state_Y

            # Compute the timeseries for each time steps
            # the loop would not be needed if we wrote a single ode function for both set of variables.
            for ind in range(0, n_timestep - 1):
                # Each timestep is computed by using a 4th order Runge-Kutta solver
                x = self._rk4ode(self._l95ode_true_X, np.array([time_steps[ind], time_steps[ind + 1]]),
                                 timeseries_X[:, ind],
                                 [timeseries_Y[:,
                                  ind]])  # we pass the value of the other set of variables as the parameter.
                y = self._rk4ode(self._l95ode_true_Y, np.array([time_steps[ind], time_steps[ind + 1]]),
                                 timeseries_Y[:, ind],
                                 [timeseries_X[:,
                                  ind]])  # we pass the value of the other set of variables as the parameter.

                timeseries_X[:, ind + 1] = x[:, -1]
                timeseries_Y[:, ind + 1] = y[:, -1]

            result_X.append(timeseries_X.flatten())
            result_Y.append(timeseries_Y.flatten())

        # return an array of objects of type Timeseries
        return result_X, result_Y

    def _l95ode_par(self, t, x, parameter):
        """
        The parameterized two-tier lorenz 95 system defined by a set of symmetic
        ordinary differential equation. This function evaluates the differential
        equations at a value x of the time-series

        Parameters
        ----------
        x: numpy.ndarray of dimension px1
            The value of timeseries where we evaluate the ODE
        parameter: Python list
            The set of parameters needed to evaluate the function
        Returns
        -------
        numpy.ndarray
            Evaluated value of the ode at a fixed timepoint
        """
        # Initialize the array containing the evaluation of ode
        dx = np.zeros(shape=(x.shape[0],))
        eta = parameter[0]
        theta = parameter[1]
        # Deterministic parameterization for fast weather variables
        # ---------------------------------------------------------
        # assumed to be polynomial, degree of the polynomial same as the
        # number of columns in closure parameter
        degree = theta.shape[0]
        X = np.ones(shape=(x.shape[0], 1))
        for ind in range(1, degree):
            X = np.column_stack((X, pow(x, ind)))

        # deterministic reparametrization term
        # ------------------------------------
        gu = np.sum(X * theta, 1)

        # ODE definition of the slow variables
        # ------------------------------------
        dx[0] = -x[-2] * x[-1] + x[-1] * x[1] - x[0] + self.F - gu[0] + eta[0]
        dx[1] = -x[-1] * x[0] + x[0] * x[2] - x[1] + self.F - gu[1] + eta[1]
        for ind in range(2, x.shape[0] - 2):
            dx[ind] = -x[ind - 2] * x[ind - 1] + x[ind - 1] * x[ind + 1] - x[ind] + self.F - gu[ind] + eta[ind]
        dx[-1] = -x[-3] * x[-2] + x[-2] * x[1] - x[-1] + self.F - gu[-1] + eta[-1]

        return dx

    def _l95ode_true_X(self, t, x, parameter):
        """
        The parameterized two-tier lorenz 95 system defined by a set of symmetic
        ordinary differential equation. This function evaluates the differential
        equations at a value x of the time-series

        Parameters
        ----------
        x: numpy.ndarray of dimension px1
            The value of timeseries where we evaluate the ODE
        parameter: Python list
            The set of parameters needed to evaluate the function
        Returns
        -------
        numpy.ndarray
            Evaluated value of the ode at a fixed timepoint
        """
        # Initialize the array containing the evaluation of ode
        dx = np.zeros(shape=(x.shape[0],))
        y = parameter[0]

        # ODE definition of the slow variables
        # ------------------------------------

        dx[0] = -x[-2] * x[-1] + x[-1] * x[1] - x[0] + self.F - self.hc_over_b * np.sum(y[0: self.J])
        dx[1] = -x[-1] * x[0] + x[0] * x[2] - x[1] + self.F - self.hc_over_b * np.sum(y[self.J: 2 * self.J])
        for ind in range(2, x.shape[0] - 2):
            dx[ind] = -x[ind - 2] * x[ind - 1] + x[ind - 1] * x[ind + 1] - x[ind] + self.F - \
                      self.hc_over_b * np.sum(y[self.J * ind: self.J * (ind + 1)])
        dx[-1] = -x[-3] * x[-2] + x[-2] * x[1] - x[-1] + self.F - self.hc_over_b * np.sum(y[-1 * self.J:])

        return dx

    def _l95ode_true_Y(self, t, y, parameter):
        """
        The parameterized two-tier lorenz 95 system defined by a set of symmetic
        ordinary differential equation. This function evaluates the differential
        equations at a value x of the time-series

        Parameters
        ----------
        y: numpy.ndarray of dimension px1
            The value of timeseries where we evaluate the ODE
        parameter: Python list
            The set of parameters needed to evaluate the function
        Returns
        -------
        numpy.ndarray
            Evaluated value of the ode at a fixed timepoint
        """
        # Initialize the array containing the evaluation of ode
        dy = np.zeros(shape=(y.shape[0],))
        x = parameter[0]

        # ODE definition of the fast variables
        # ------------------------------------
        for ind in range(y.shape[0] - 3):
            dy[ind] = - self.cb * y[ind + 1] * (y[ind + 2] - y[ind - 1]) - self.c * y[ind] + \
                      self.hc_over_b * x[ind // self.J]  # // for the integer division.

        dy[-2] = - self.cb * y[- 1] * (y[0] - y[-3]) - self.c * y[-2] + \
                 self.hc_over_b * x[-2 // self.J]  # // for the integer division.

        dy[-1] = - self.cb * y[0] * (y[1] - y[-2]) - self.c * y[-1] + \
                 self.hc_over_b * x[-1 // self.J]  # // for the integer division.

        return dy

    def _rk4ode(self, ode, timespan, timeseries_initial, parameter):
        """
        4th order runge-kutta ODE solver.

        Parameters
        ----------
        ode: function
            The function defining Ordinary differential equation
        timespan: numpy.ndarray
            A numpy array containing the timepoints where the ode needs to be solved.
            The first time point corresponds to the initial value
        timeseries_initial: np.ndarray of dimension px1
            Intial value of the time-series, corresponds to the first value of timespan
        parameter: Python list
            The parameters needed to evaluate the ode
        Returns
        -------
        np.ndarray
            Timeseries initiated at timeseries_init and satisfying ode solved by this solver.
        """

        timeseries = np.zeros(shape=(timeseries_initial.shape[0], timespan.shape[0]))
        timeseries[:, 0] = timeseries_initial

        for ind in range(0, timespan.shape[0] - 1):
            time_diff = timespan[ind + 1] - timespan[ind]
            time_mid_point = timespan[ind] + time_diff / 2
            k1 = time_diff * ode(timespan[ind], timeseries_initial, parameter)
            k2 = time_diff * ode(time_mid_point, timeseries_initial + k1 / 2, parameter)
            k3 = time_diff * ode(time_mid_point, timeseries_initial + k2 / 2, parameter)
            k4 = time_diff * ode(timespan[ind + 1], timeseries_initial + k3, parameter)
            timeseries_initial = timeseries_initial + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            timeseries[:, ind + 1] = timeseries_initial
        # Return the solved timeseries at the values in timespan
        return timeseries


class StochLorenz95_with_statistics(StochLorenz95):

    def __init__(self, parameters, initial_state=None, F=10, b=10, h=1, c=4, J=8, K=40, time_units=4,
                 n_timestep_per_time_unit=30, moments=3, auto_covariances=5, covariances=5, cross_covariances=5,
                 degree=1, cross=False, name='StochLorenz95'):
        self.statistics = LorenzLargerStatistics(moments=moments, auto_covariances=auto_covariances,
                                                 covariances=covariances, cross_covariances=cross_covariances,
                                                 degree=degree, cross=cross)
        self.out_dim = moments + auto_covariances + covariances + 2 * cross_covariances

        super().__init__(parameters, initial_state=initial_state, F=F, b=b, h=h, c=c, J=J, K=K, time_units=time_units,
                         n_timestep_per_time_unit=n_timestep_per_time_unit, name=name)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        output = super().forward_simulate(input_values, k, rng=rng)
        result = self.statistics.statistics(output)
        return [x for x in result]  # need to convert back to list otherwise ABC does not work there.

    def get_output_dimension(self):
        return self.out_dim
