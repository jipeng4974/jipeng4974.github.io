# On NCO

> Non-convex optimization, more like art

---

LLMS index: [llms.txt](/llms.txt)

---

Convex optimization generally converges in polynomial time; linear programming and least squares are special cases of convex optimization.

Non-convex optimization (NCO) is a class of problems that are at least NP-hard, with no general-purpose solution. Determining whether a problem has a solution, whether a local optimum is the global optimum, or whether the objective function is bounded all blow up exponentially with the number of variables and constraints. Local optimization methods are sensitive to algorithm parameters and heavily dependent on the initial guess, which makes local non-convex optimization more art than technology — by comparison, linear programming has no art to it at all.

As universal function approximators, the most important role of deep neural networks is fitting non-convex functions, because complex problems generally cannot be fit by convex functions. Generative models like ChatGPT are essentially non-convex optimization of the mutual information between target and input. How to train a model well is still an art today.

Stochastic gradient descent (SGD) has been proven to converge on convex, differentiable, and Lipschitz-continuous functions, but its effectiveness on non-convex functions is still undetermined. SGD converges slowly, is not guaranteed to reach even a local optimum, let alone the global one. If you pick a point close enough to the global optimum, SGD might converge to it, but that is time-consuming on the one hand and only works in special cases on the other. For deep neural networks, once you fall into a wrong local optimum, you have to try different initialization configurations or add extra noise to the gradient updates. If you hit a saddle point, you need to find the Hessian matrix or compute a descent direction. If you get stuck in a low-gradient region, you need batchnorm, or ReLU as the activation function. If high curvature makes the steps too large, you should use adaptive step sizes or limit the magnitude of gradient steps. On top of that, if the hyperparameters are off, you need various hyperparameter optimization methods. In short, NCO for deep learning is still at the art stage.
