# Language Model

- [Language Model](#language-model)
  - [Autoregressive Language Model](#autoregressive-language-model)
  - [Autoencoding Language Model](#autoencoding-language-model)
  - [BLEU](#bleu)
  - [Transformer](#transformer)
    - [RNN](#rnn)
    - [Attention](#attention)

## Autoregressive Language Model

自回归模型即在给定之前的所有token，输出下一个token是什么(指利用上文信息或者下文信息)，是单向的。给定源语句 $(x_1, \cdots, x_m)$ ，目标语句 $(y_1, \cdots, y_n)$ 是按照如下的方式生成的:

$$p(y|x) = \prod_{t}{p(y_t|y_{\lt t},x)}$$

$t$ 是当前的时刻， $y_{\lt t}$ 是当前时刻已经生成的token，由于前后的依赖关系，AR模型的生成往往需要 $O(n)$ 次循环。AR模型适合用于自然语言生成(NLG)任务。GPT是典型的自回归模型，缺点是生成速度慢，non-autoregressive模型就是想要减少生成时的循环次数。

## Autoencoding Language Model

自编码模型是把输入token打乱，学习出一个中间表示(隐空间)，再用这个中间表示还原出原始的序列。BERT作为典型的AE模型就是通过mask掉一部分token，再重建恢复原始序列。AE模型能较好地编码上下文信息，因此擅长自然语言理解(NLU)的任务。

## BLEU

BLEU全称是 *Bilingual Evaulation Understudy*, 意思是双语评估替补。它是一种评估机器翻译质量的方法，也可以推广到其他NLP问题中。

假设参考翻译集合为: 

```
The cat is on the desk.
There is a cat on the desk.
```

那么很自然想到，用`实际翻译结果中出现在参考翻译中的单词数`除以`实际翻译结果单词总数`，来评估结果的好坏。例如，若翻译结果为`The cat are on the desk`。则评分为：`5/6`，只有`are`没有出现，这看起来是合理的。但是若翻译结果为`is is is is is is is`，那么很显然，评分为`6/6`，因为`is`在参考翻译句子中出现了。很明显，这这种方案不合理。

错误出现在对单词的计数不合理，一个解决方法是，我们规定`实际翻译结果中每个单词的计数`不得超过在`单个参考翻译中出现的最大次数`。在上述`is is is is is is`翻译结果中，单词`is`在参考翻译中出现的最大次数是`1`，因此，只能被记`1`次，评分为`1/6`。这是比较合理的。

还有个因素需要考虑，假如实际翻译句子为`desk the on cat a is there`，那么得分为`7/7`，虽然单词都出现了，句子的流畅度却没有考虑。因此，根据“平滑”的思想，进一步考虑`1-gram`到`4-gram`。具体来说：我们除了对单个单词计数，还对2、3、4个单词组成的词组进行计数。$n = 1,2,3,4$ 时，每 $n$ 个单词为一组，设实际翻译中每个元素为 $x_i^n$ , 则有:

$$score_n = \sum_i{x_i^n在参考翻译中出现的最大次数}\quad / \quad \sum_i{x_i^n在实际翻译中出现的次数}$$

paper中的BLEU一般取为:

$$BLEU = \exp \left(\sum_{n=1}^4 score_n \right)$$

最大值时四个 $score$ 均为 $1$ , $BLEU_{max} = e^4 \approx 54.598$ .

## Transformer

### RNN



### Attention