import numpy as np

from scipy.optimize._optimize import (
    _wrap_scalar_function_maxfun_validation,
    _MaxFuncCallError,
    OptimizeResult,
    _call_callback_maybe_halt,
    _status_message,
)


def _minimize_neldermead(
    func,
    x0,
    args=(),
    callback=None,
    maxiter=None,
    maxfev=None,
    disp=False,
    return_all=False,
    initial_simplex=None,
    xatol=1e-4,
    fatol=1e-4,
    adaptive=False,
    bounds=None,
):
    maxfun = maxfev
    retall = return_all

    x0 = np.atleast_1d(x0).flatten()
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2 / dim
        psi = 0.75 - 1 / (2 * dim)
        sigma = 1 - 1 / dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError(
                "Nelder Mead - one of the lower bounds "
                "is greater than an upper bound."
            )
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            print("Initial guess is not within the specified bounds")

    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt) * y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    if bounds is not None:
        # The default simplex construction may make all entries (for a given
        # parameter) greater than an upper bound if x0 is very close to the
        # upper bound. If one simply clips the simplex to the bounds this could
        # make the simplex entries degenerate. If that occurs reflect into the
        # interior.
        msk = sim > upper_bound
        # reflect into the interior
        sim = np.where(msk, 2 * upper_bound - sim, sim)
        # but make sure the reflection is no less than the lower_bound
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)

    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while fcalls[0] < maxfun and iterations < maxiter:
        try:
            if (
                np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol
                and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol
            ):
                break

            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            fxr = func(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                fxe = func(xe)

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        fxc = func(xc)

                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        fxcc = func(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                            fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        finally:
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
            if retall:
                allvecs.append(sim[0])
            intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])
            if _call_callback_maybe_halt(callback, intermediate_result):
                break

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message["maxfev"]
        if disp:
            # warnings.warn(msg, RuntimeWarning, stacklevel=3)
            print(msg)
            # warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message["maxiter"]
        if disp:
            print(msg)
            # warnings.warn(msg, RuntimeWarning, stacklevel=3)
    else:
        msg = _status_message["success"]
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(
        fun=fval,
        nit=iterations,
        nfev=fcalls[0],
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=x,
        final_simplex=(sim, fsim),
    )
    if retall:
        result["allvecs"] = allvecs
    return result


# Define the objective function
def objective_function(vector):
    x, y, z = vector
    return (x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2


# Initial guess
initial_guess = [0.0, 0.0, 0.0]

result = _minimize_neldermead(objective_function, initial_guess)
print(result)
