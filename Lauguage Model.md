# Language Model

## Autoregressive Language Model

自回归模型即在给定之前的所有token，输出下一个token是什么(指利用上文信息或者下文信息)，是单向的。给定源语句 $(x_1, \cdots, x_m)$ ，目标语句 $(y_1, \cdots, y_n)$ 是按照如下的方式生成的:

$$p(y|x) = \prod_{t}{p(y_t|y_{\lt t},x)}$$

$t$ 是当前的时刻， $y_{\lt t}$ 是当前时刻已经生成的token，由于前后的依赖关系，AR模型的生成往往需要 $O(n)$ 次循环。AR模型适合用于自然语言生成(NLG)任务。GPT是典型的自回归模型，缺点是生成速度慢，non-autoregressive模型就是想要减少生成时的循环次数。

## Autoencoding Language Model

自编码模型是把输入token打乱，学习出一个中间表示(隐空间)，再用这个中间表示还原出原始的序列。BERT作为典型的AE模型就是通过mask掉一部分token，再重建恢复原始序列。AE模型能较好地编码上下文信息，因此擅长自然语言理解(NLU)的任务。