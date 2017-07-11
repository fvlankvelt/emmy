import breeze.linalg._
import org.scalatest.FlatSpec

class LinearRegressionSpec extends FlatSpec {

  import Variable._
  import Function._

  "A model" should "contain a distribution" in {
    implicit val model = new Model

    val X = VectorVariable(2)

    val a = Normal(mu=0.0f, sigma=10.0f)
    val b = Normal(mu=DenseVector(0.0f, 0.0f), sigma=DenseVector(10.0f))
    val Y = a + sum(b * X)

    val approximation = model.fit(Map(
      X -> DenseVector(1.0f, 0.2f),
      Y -> 2.0f
    ))
  }

}
