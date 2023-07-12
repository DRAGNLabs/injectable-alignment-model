## Evaluation Notes

**Paper:** [Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models](https://arxiv.org/pdf/2206.04615.pdf)

Beyond the Imitation Game Benchmark (BIG-bench) is a (Google affiliated?) paper with over 400 authors from many different fields designing 
a benchmark with a human baseline. The purpose is to distinguish more exactly what LMs *can* and *cannot* do; 209 tasks are developed.  

**Paper:** [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them](https://arxiv.org/pdf/2210.09261.pdf)

Stemming from that paper comes Google-Stanford's Challenging BIG-Bench Tasks paper in the which they introduce 23 tasks that BIG-bench underperformed on compared to human raters. These are passed through other filters, listed below. 

| Task Count | Criteria |
|---|---|
| 209 | All BIG-Bench |
| 187 | After filtering out tasks (AFOT) w/ more than 3 subtasks |
| 130 | AFOT w/ <103 examples (3 for few-shot, 100 for evaluation)|
| 85 | AFOT w/out human-rater baselines |
| 78 | AFOT w/out discrete answers |
| 36 | AFOT where best model beats avg. human score|
| 23 | AFOT extremely difficult tasks that are outside the scope of this work |
| Final = 23 | BIG-Bench Hard Tasks (BBH) |

*NOTE:* If ORCA is only evaluated on exact-match benchmarks(?), can it be said to follow CoT well?
