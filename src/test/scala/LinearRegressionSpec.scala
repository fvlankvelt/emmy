import breeze.linalg._
import org.scalatest.FlatSpec
import pp._

class LinearRegressionSpec extends FlatSpec {

  import pp.Function._

  "A model" should "contain a distribution" in {
    implicit val model = new Model

    val X = VectorVariable(2)

    val a = Normal(mu = 3.0f, sigma = 10.0f)
    val b = Normal(mu = DenseVector(0.0f, 0.0f), sigma = DenseVector(10.0f, 10.0f))
    val sigma = Gamma(1.0f, 1.0f)

//    val mu : ScalarVariableLike = a + sum(b * X)
    val mu : ScalarVariableLike = sum(b * X)
    val Y = Normal(mu = mu, sigma = sigma)
//    val logp = Y.logp() + (a.logp() + b.logp() + sigma.logp())
    val logp = Y.logp()

    val context = model.context
      .copy(variables = Seq(
        Assignment(X, DenseVector(1.0f, 2.0f)),
        Assignment(Y, 0.5f),
        Assignment(sigma, 0.5f),
        Assignment(a, 1.0f),
        Assignment(b, DenseVector(1.0f, 0.5f))
      ))

    val logpVal = logp.eval(context)

//    val dYda = logp.grad(a)
//    assert(dYda.isDefined)
//    val dYdaVal = context.eval(dYda.get)
//    assert(dYdaVal == 65.0398f)

//    assert(mu.grad(b).isDefined)
//    assert(Y.logp().grad(mu).isDefined)

    val dYdb = logp.grad(b)
    assert(dYdb.isDefined)
    val dYdbVal = context.eval(dYdb.get)
    assert(dYdbVal == DenseVector(1.0f, 2.0f))

    val approximation = model.fit(Seq(
      Map(
        X -> DenseVector(1.0f, 0.2f),
        Y -> 2.0f
      ),
      Map(
        X -> DenseVector(0.2f, 2.0f),
        Y -> 1.0f
      )
    ))
  }

}
