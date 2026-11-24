# ODE 采样器。torchdyn 只在真正调用 ODE 采样时才会导入，
# 因此主训练/默认 Euler 采样不需要安装 torchdyn。

import torch


def _build_solver(solver_name, dtype):
    import torchdyn.numerics.solvers.ode as solvers

    class Euler(solvers.Euler):
        pass

    class Midpoint(solvers.Midpoint):
        pass

    class RungeKutta4(solvers.RungeKutta4):
        def step(self, **kwargs):
            self.sync_device_dtype(kwargs["x"], kwargs["t"])
            return super().step(**kwargs)

    class ImplicitEuler(solvers.ImplicitEuler):
        def step(self, **kwargs):
            self.sync_device_dtype(kwargs["x"], kwargs["t"])
            return super().step(**kwargs)

    solver_dict = {
        "euler": Euler,
        "midpoint": Midpoint,
        "rk4": RungeKutta4,
        "rk-4": RungeKutta4,
        "RungeKutta4": RungeKutta4,
        "ieuler": ImplicitEuler,
        "implicit_euler": ImplicitEuler,
    }
    return solver_dict[solver_name](dtype)


def str_to_solver(solver_name, dtype=torch.float32):
    """把字符串转换成 torchdyn solver；调用前需要安装可选依赖 torchdyn。"""
    return _build_solver(solver_name, dtype)
