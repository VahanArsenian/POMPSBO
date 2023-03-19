# Reviewer My6r
First, we would like to thank the reviewer for their feedbacks on the paper and for seeing the potential of the presented method. We try to address the main questions below:

## Type of interventions
An intervention is a fundamental procedure in our paper. When we say that a variable $X$ is intervened or controlled according to a policy $\pi(X\mid \mathbf{C})$ it implies that the variable now depends on variables in $\mathbf{C}$ instead of its previous parents in the causal graph, and that the functional form of that dependency is given by the policy $\pi$. 

[//]: # (%All incoming edges of variable $X$ in causal graph $\mathcal{G}$ are deleted, and new ones are introduced from $\forall C\in\mathbf{C}$ and a new structural equation of $X$ is introduced according to the policy when it is said to be controlled by a policy $\pi&#40;X\mid \mathbf{C}&#41;$. )
Moreover, we assume that no other variable is affected during this surgery both parametrically and structurally. Hence, our interventions are assumed to be precise and not fat hand ones [1, 2].

## POMPS as a dimension of BO
The idea of optimizing POMPSs as a dimension of BO is attractive at first glance. This would require defining a kernel over POMPS (potentially through some POMPS encoding). Furthermore, the space of optional functions is not smooth with respect to POMPS. Even if two POMPSs are only slightly different in terms of interventional and contextual variables, the space of policies that they induce may be drastically different, leading to very different optimal policies from one POMPS to the other.

We have actually considered such an approach during the initial stages of our research, but given the difficulty to find a practical kernel over POMPSs that would allow to model our objective function, we abandoned this idea.

## Scalability with respect to number of POMPS and number of variables.
At the time of this research, there is no known method to generate all the POMPSs of a given causal graph without iterating over all MPSs. 
As the cardinality of the set of MPSs scales exponentially with the number of variables, POMPS identification also suffers an exponential computational cost with respect to the number of variables (note that this cost is not related to the sample complexity of the BO as this is pure analysis of the causal graph structure). 
Nevertheless, we endeavoured to make POMPS identification as efficient as possible. 
A sequence of operations is performed on MPSs to identify the POMPSs, and we perform these operations in increasing complexity order so as to perform the most expensive operations on the smallest sets of MPSs. 
Moreover, those operations, such as checking acyclicity of the causal graph under an MPS, are embarrassingly parallel. 
We completely understand and emphasize the importance of more efficient algorithms to find the set of POMPSs and leave it to the future, as the current work focuses on practical convergence properties in terms of sample complexity.

Remark: as underlined and illustrated in our paper, even though POMPS identification limit the scalability of the proposed method, existing alternative methods that would consist in omitting the causal structure or omitting exploring some POMPSs are exposed to the risk of totally failing to identify the best policy and best policy scope.


#### Additional remarks
- "Const. C" in Figure 1.: it simply means that the objective is  $Y = X_2 U_2 + \lambda \times C$ where $\lambda \in \mathbb{R}$ is some constant.

- Missing definition of X(S) in Definition 3.3: X(S) is introduced in line 209, and it designates the set of interventional
variables for an MPS S.

- Definition 3.4: shouldn't S subsume S' and not the opposite?:  we say that S' is subsumed by S, so the reviewer is right (and we can rewrite it in active voice to make it clearer): S subsumes S'.

-  $X_1$ based on $X_1$: the reviewer is right, it should be  $X_2$ based on $X_1$.

- We will make sure to eliminate all remaining typos.

## References 

[1] Eronen, M. I. (2020). Causal discovery and the problem of psychological interventions. New Ideas in Psychology, 59, 100785. https://doi.org/10.1016/j.newideapsych.2020.100785

[2] Glymour, M. M., & Spiegelman, D. (2017). Evaluating Public Health Interventions: 5. Causal Inference in Public Health Research—Do Sex, Race, and Biological Factors Cause Health Outcomes? American Journal of Public Health, 107(1), 81–85. https://doi.org/10.2105/ajph.2016.303539

# Reviewer ELCc
First, we would like to thank the reviewer for their valuable feedbacks. We try to provide an answer to the main questions and remarks below, and notably to the existence of applications to the proposed method, and to the role of HEBO.

# Real world applications
Our method assumes that the causal graph of the optimization problem is known *a priori*. We acknowledge that this assumption may be restrictive and limit the applicability of the proposed method. However, we hope our method is applicable in well-studied domains such as physics, where a causal graph can be constructed from domain knowledge. Similarly, domain knowledge is also available in ecology. For example, the research on scleractinian coral and reef-scale calcification [1] provides a causal graph for coral growth, and our method can be used to optimize the growth rate of corals if, of course, experimentation is allowed; otherwise, optimization is irrelevant and one may only observe. The example of optimizing the paper's Prostate-Specific Antigen (PSA) is also taken from a real-world study [2]. Another domain that has been using the notion of causation is sociology. The study of earnings inequality in academe [3] provides a causal graph for variables that may affect salary in an academic environment.

Domain knowledge is not the only way of constructing causal graphs. The vast literature on causal discovery [4,5,6] provides the toolbox for constructing such graphs from observational and interventional data. Furthermore, it is proven that in the worst case for a set of causally insufficient variables, the number of necessary structural interventions grows linearly with the number of variables.

Another direction is text-based causal graph learning. People have documented their discoveries in corpora, and those volumes of text contain causal knowledge encoded within them [7]. For example, Electronic Medical Record (EMR) is a promising source of such knowledge, and it has been successfully utilized [8] in constructing causal graphs for various diseases.

Our paper aims to highlight issues in the current methods and demonstrate that CoCaBO is a practical solution to those. This requires a large number of evaluations in various setups, which is hard in real-world setups, and simulators have been used, some mimicking real-world examples. CaBO also reports results on simulators due to the same reasons. However, we acknowledge the importance of real-world evaluations, and we leave it as future work.

## Use of HEBO and details on experimental setup
We utilize HEBO [9] not only for the proposed method but also for the other benchmarks, so both CoBO and CaBO benefit from all functionalities of HEBO, which makes the comparison fair.
All methods use the same kernel, namely the Matern kernel [10], over the product space of contextual and interventional variables.
Although CaBO uses only one acquisition function in the original paper, we see that it benefits from multi-acquisition function optimization implemented in HEBO in all setups.

Nevertheless, to address the reviewer's concern that HEBO may play a paramount role in the method comparisons, 
we conduct an extra experiment where we replace HEBO with a standard BO method for CoCa-BO. 
We show that qualitative analysis is not impacted by the deactivation of HEBO's tricks (CoCa-BO suffers lower regrets than CoBO and CaBO), but the use of HEBO increases the sample efficiency of CoCa-BO.

## References
[1] Courtney, T. A., Lebrato, M., Bates, N. R., Collins, A., De Putron, S. J., Garley, R., Johnson, R., Molinero, J. C., Noyes, T. J., Sabine, C. L., & Andersson, A. J. (2017). Environmental controls on modern scleractinian coral and reef-scale calcification. Science Advances, 3(11). https://doi.org/10.1126/sciadv.1701356

[2] Ferro, A., Pina, F., Severo, M., Dias, P., Botelho, F., & Lunet, N. (2015). Use of statins and serum levels of Prostate Specific Antigen. Acta Urológica Portuguesa. https://doi.org/10.1016/j.acup.2015.02.002

[3] Leahey, E. (2007). Not by Productivity Alone: How Visibility and Specialization Contribute to Academic Earnings. American Sociological Review, 72(4), 533–561. https://doi.org/10.1177/000312240707200403

[4] Spirtes, P., Glymour, C., & Scheines, R. (2012). Causation, Prediction, and Search. Springer Science & Business Media.

[5] Glymour, C., Zhang, K., & Spirtes, P. (2019). Review of Causal Discovery Methods Based on Graphical Models. Frontiers in Genetics, 10. https://doi.org/10.3389/fgene.2019.00524

[6] Kocaoglu, M., Shanmugam, K., & Bareinboim, E. (2017b). Experimental Design for Learning Causal Graphs with Latent Variables. Neural Information Processing Systems, 30, 7018–7028. https://papers.nips.cc/paper/7277-experimental-design-for-learning-causal-graphs-with-latent-variables.pdf

[7] Friedman, S. L., Magnusson, I. H., Sarathy, V., & Schmer-Galunder, S. (2022). From Unstructured Text to Causal Knowledge Graphs: A Transformer-Based   Approach. ArXiv (Cornell University). https://doi.org/10.48550/arxiv.2202.11768

[8] Nordon, G., Koren, G., Shalev, V., Kimelfeld, B., Shalit, U., & Radinsky, K. (2019). Building Causal Graphs from Medical Literature and Electronic Medical Records. Proceedings of the. AAAI Conference on Artificial Intelligence, 33(01), 1102–1109. https://doi.org/10.1609/aaai.v33i01.33011102

[9] Cowen-Rivers, A. I., Lyu, W., Tutunov, R., Wang, Z., Grosnit, A., Griffiths, R. R., Maraval, A. M., Jianye, H., Wang, J., Peters, J., & Ammar, H. B. (2020). HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation. ArXiv (Cornell University). https://doi.org/10.48550/arxiv.2012.03826

[10] Rasmussen, C. E., & Williams, C. K. I. (2005). Gaussian Processes for Machine Learning. MIT Press.


# Reviewer 4j97

We thank the reviewer for their constructive comments and we hope that the elements below will shed light on the points that remain to be clarified.

## Benchmark discrepancy
We would like to make a precision regarding the first weakness mentioned in the review: the fact that "the used banchmarks [baselines ?] were not designed to solve the same problem and thus they do not exploit causal knowledge to find the optimum".

We note that the baseline CaBO actually do assume access to a causal graph, but we show that it does not fully exploit it. Indeed, we demonstrate that CoCa-BO outperforms CaBO by utilizing context. Additionally, we demonstrate that the direct application of CoBO, a well-established method for contextual optimization,  when a causal graph is given can lead to suffering linear regret.

