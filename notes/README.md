# notes

- Deepmind blog post on modern reinfocement learning [here](https://deepmind.com/blog/article/Agent57-Outperforming-the-human-Atari-benchmark)
- ApeX paper describes distributed reinforcement learning with gradient prioritization and replay pooling service, [here](https://openreview.net/pdf?id=H1Dy---0Z).
- A meta-learning summary, [here](https://lilianweng.github.io/lil-log/2019/06/23/meta-reinforcement-learning.html). 
- Online learning for stochastic quasi-newton methods, [here](http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf)
- Example of separating gradient calculation and application:
```
grads = autograd.grad(loss, vars_list)
for p, g in zip(vars_list, grads):
    p.grad.fill_(g)
optimizer.step()
```

