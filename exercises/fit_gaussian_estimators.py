from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(samples)
    print("(mu, var) = (", univariate_gaussian.mu_, ", ", univariate_gaussian.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1001, 10)
    distance_from_mu = []
    for size in sample_sizes:
        univariate_gaussian.fit(samples[:size])
        distance_from_mu.append(abs(univariate_gaussian.mu_ - mu))

    go.Figure([go.Scatter(x=sample_sizes, y=distance_from_mu, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated- And True Value Of The"
                                     r" Expectation, As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="$d\\text{ - distance from mu}$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_gaussian.fit(samples)
    pdf_values = univariate_gaussian.pdf(samples)

    go.Figure([go.Scatter(x=samples, y=pdf_values, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Probability Density Function Of Sample Values}$",
                               xaxis_title="$m\\text{ - sample values}$",
                               yaxis_title="$d\\text{ - pdf}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma =            [[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples)
    print("Estimated Expectation Vector is: ", multivariate_gaussian.mu_)
    print("Estimated Covariance Matrix is: \n", multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    log_like_matrix = np.zeros((f.size, f.size))
    for f1 in range(f.size):
        for f3 in range(f.size):
            mu = [f[f1], 0, f[f3], 0]
            val = MultivariateGaussian.log_likelihood(mu, sigma, samples)
            log_like_matrix[f1, f3] = val
    go.Figure([go.Heatmap(x=f, y=f, z=log_like_matrix)], layout=go.Layout(
                        title=r"$\text{Maximum likelihood of Models as Function of Changes in Expectation}$",
                        xaxis_title="$f3\\text{ - Estimated Expectation at 3-rd Coordinate}$",
                        yaxis_title="$f1\\text{ - Estimated Expectation at 1-st Coordinate}$")).show()

    # Question 6 - Maximum likelihood
    print("The model that achieved Maximum likelihood is: (f1, f3) = (",
          f[np.argmax(log_like_matrix, axis=0)][0], ", ", f[np.argmax(log_like_matrix, axis=1)][0],
          ")\nMaximum likelihood is: ", np.amax(log_like_matrix))


if __name__ == '__main__':
    np.random.seed(0)

    test_univariate_gaussian()
    test_multivariate_gaussian()





