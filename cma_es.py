import numpy as np


def _default_popsize(n):
    return 4 + int(3 * np.log(max(n, 1)))


class CMAES:

    def __init__(self, objective, x0, sigma0=0.5, bounds=None, popsize=None,
                 max_generations=30, seed=None):
        self.objective = objective
        self.n = n = len(x0)
        self.bounds = bounds
        self.max_generations = max_generations

        self.popsize = popsize if popsize is not None else _default_popsize(n)
        self.mu = max(1, self.popsize // 2)

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        w = w / w.sum()
        self.w = w
        self.mueff = 1.0 / (w ** 2).sum()

        n = float(n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.mean = np.array(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.rng = np.random.default_rng(seed)

        self.generation = 0
        self.best_x = self.mean.copy()
        self.best_f = np.inf
        self.history = []

    def _sample_one(self, B, D):
        z = self.rng.standard_normal(self.n)
        y = B @ (D * z)
        x = self.mean + self.sigma * y
        if self.bounds is not None:
            x = np.clip(x, self.bounds[0], self.bounds[1])
        return x

    def _objective_safe(self, x):
        f = self.objective(x)
        return np.inf if f is None or np.isnan(f) else float(f)

    def run(self):
        n = self.n
        for _ in range(self.max_generations):
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 0.0, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            pop = []
            for _ in range(self.popsize):
                x = self._sample_one(B, D)
                f = self._objective_safe(x)
                pop.append((f, x))
            pop.sort(key=lambda t: t[0])
            pop = [(f, x) for f, x in pop if np.isfinite(f)]
            if not pop:
                break
            xs = np.stack([x for _, x in pop[:self.mu]])

            if pop[0][0] < self.best_f:
                self.best_f = pop[0][0]
                self.best_x = pop[0][1].copy()

            m_old = self.mean.copy()
            self.mean = (self.w[:self.mu][:, None] * xs).sum(axis=0)

            y_w = (self.mean - m_old) / self.sigma
            invsqrtC_y = B @ ((B.T @ y_w) / D)
            self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * invsqrtC_y
            hsig_denom = np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1)))
            hsig = (np.linalg.norm(self.ps) / max(hsig_denom, 1e-30) / self.chiN) < (1.4 + 2.0 / (self.n + 1.0))

            self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w

            self.C = (1.0 - self.c1 - self.cmu) * self.C \
                + self.c1 * (np.outer(self.pc, self.pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C)
            for i in range(self.mu):
                y_i = (pop[i][1] - m_old) / self.sigma
                self.C += self.cmu * self.w[i] * np.outer(y_i, y_i)

            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1.0))
            self.sigma = max(self.sigma, 1e-12)

            self.generation += 1
            self.history.append({
                'gen': self.generation, 'best_f': self.best_f,
                'mean': self.mean.copy(), 'sigma': self.sigma,
            })

        return self.best_x, self.best_f


def minimize(objective, x0, sigma0=0.5, bounds=None, popsize=None,
             max_generations=30, seed=None):
    solver = CMAES(objective, x0, sigma0=sigma0, bounds=bounds, popsize=popsize,
                   max_generations=max_generations, seed=seed)
    best_x, best_f = solver.run()
    return best_x, best_f, solver.history


if __name__ == '__main__':
    def _test(name, f, x0, sigma0, bounds, target, tol=1e-2):
        best_x, best_f, hist = minimize(f, x0, sigma0=sigma0, bounds=bounds,
                                        max_generations=60, seed=0)
        ok = abs(best_f - target) < tol
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: best_f={best_f:.6f} "
              f"(target {target}) best_x={np.round(best_x, 4)}")
        return ok

    rng = np.random.default_rng(0)
    _test("sphere-5d", lambda x: float(np.sum(x ** 2)), np.zeros(5), 0.5, None, 0.0)
    _test("rosenbrock-2d", lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2),
          np.array([-1.0, 1.0]), 0.3, None, 0.0, tol=1e-1)
    _test("rastrigin-3d", lambda x: float(30 + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))),
          np.full(3, 0.5), 1.0, [-5.12, 5.12], 0.0, tol=1e-1)
    _test("bounded-quad", lambda x: float((x[0] - 100.0) ** 2 + (x[1] + 50.0) ** 2),
          np.array([0.0, 0.0]), 30.0, [-1000.0, 1000.0], 0.0, tol=1e-1)
    print("CMA-ES self-test complete.")
