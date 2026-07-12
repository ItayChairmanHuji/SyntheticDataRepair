\begin{small}
\smallskip
\hrule
\revised{%
\begin{align}
\textbf{min}\ \ & \alpha \cdot \frac{n - \sum_{l=1}^{n} x_l}{n} \;+\; \frac{1-\alpha}{|\marginalset|}\sum_{m \in \marginalset} \frac{d_m}{\tmarg{D_p}{a_1}{a_2}} \label{ilp:global}\\
\textbf{s.t.}\ \ & x_i + x_j \leq 1, \quad \forall\, t_i,t_j \text{ conflicting on } \constraints \nonumber\\
& \textstyle\sum_{l} z_{ml} \geq \!\!\!\sum_{\substack{l:\, t_l[A_i]=a_1,\\ t_l[A_j]=a_2}}\!\!\!\!\!~~~~ x_l \;-\; \tmarg{D_p}{a_1}{a_2}\!\!\sum_{l}x_l,\ \forall m \in \marginalset \nonumber\\
& \textstyle\sum_{l} z_{ml} \geq \tmarg{D_p}{a_1}{a_2}\!\!\sum_{l}x_l \;- ~~\!\!\!\! \sum_{\substack{l:\, t_l[A_i]=a_1,\\ t_l[A_j]=a_2}}\!\!\!\!\! ~~~~x_l,\ \forall m \in \marginalset \nonumber\\
& z_{ml} \leq d_m,\quad z_{ml} \leq x_l,\quad z_{ml} \geq d_m-(1-x_l),\ \forall m,l \nonumber\\
& d_m \geq \tmarg{D_p}{a_1}{a_2}\bigl(1 - \textstyle\sum_{l} x_l\bigr),\ \forall m \in \marginalset \nonumber\\
& d_m \geq 0,\ \ z_{ml} \geq 0,\ \ x_l \in \{0,1\}, \quad \forall\, m \in \marginalset,\ l \in [1,n] \nonumber
\end{align}}
\hrule
\smallskip
\end{small}