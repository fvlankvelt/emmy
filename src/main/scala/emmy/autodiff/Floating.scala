package emmy.autodiff

import scala.util.Random

trait Floating[V] extends Fractional[V] {

  def sqrt: UnaryValueFunc[V]

  def log: UnaryValueFunc[V]

  def exp: UnaryValueFunc[V]

  def lgamma: UnaryValueFunc[V]

  def tanh: UnaryValueFunc[V]

  def sum: CollectValueFunc[V]

  def rnd: V
}

object Floating {

  implicit val doubleFloating: Floating[Double] = new Floating[Double] {

    override def sqrt = new UnaryValueFunc[Double] {

      val name = "sqrt"

      def apply(x: Double) = scala.math.sqrt(x)

      def grad(x: Double) = 0.5 / scala.math.sqrt(x)

    }

    override def log = new UnaryValueFunc[Double] {

      val name = "log"

      def apply(x: Double) = scala.math.log(x)

      def grad(x: Double) = 1.0 / x
    }

    override def exp = new UnaryValueFunc[Double] {

      val name = "exp"

      def apply(x: Double) = scala.math.exp(x)

      def grad(x: Double) = apply(x)
    }

    override def lgamma = new UnaryValueFunc[Double] {

      val name = "lgamma"

      override def apply(value: Double) =
        breeze.numerics.lgamma(value)

      override def grad(value: Double) =
        breeze.numerics.digamma(value)
    }

    override def tanh = new UnaryValueFunc[Double] {

      val name = "tanh"

      override def apply(value: Double) =
        breeze.numerics.tanh(value)

      override def grad(value: Double) =
        1.0 / scala.math.pow(breeze.numerics.cosh(value), 2)
    }

    override def sum = new CollectValueFunc[Double] {

      val name = "sum"

      override def apply(a: Double, b: Double) = a + b

      override val start = 0.0

      override def grad(a: Double, b: Double) = 1.0
    }

    override def rnd = Random.nextGaussian()

    override def div(x: Double, y: Double) = x / y

    override def plus(x: Double, y: Double) = x + y

    override def minus(x: Double, y: Double) = x - y

    override def times(x: Double, y: Double) = x * y

    override def negate(x: Double) = -x

    override def fromInt(x: Int) = x.toDouble

    override def toInt(x: Double) = x.toInt

    override def toLong(x: Double) = x.toLong

    override def toFloat(x: Double) = x.toFloat

    override def toDouble(x: Double) = x

    override def compare(x: Double, y: Double) = x.compareTo(y)
  }

  implicit val intFloating: Floating[Int] = new Floating[Int] {

    private def invalidOp(name: String) =
      throw new UnsupportedOperationException(s"Operation $name is not supported on Int")

    override def sqrt = invalidOp("sqrt")

    override def log = invalidOp("log")

    override def exp = invalidOp("exp")

    override def lgamma = invalidOp("lgamma")

    override def tanh = invalidOp("tanh")

    override def sum = new CollectValueFunc[Int] {

      val name = "sum"

      override def apply(a: Int, b: Int) = a + b

      override val start = 0

      override def grad(a: Int, b: Int) = 1
    }

    override def rnd = Random.nextInt()

    override def div(x: Int, y: Int) = x / y

    override def plus(x: Int, y: Int) = x + y

    override def minus(x: Int, y: Int) = x - y

    override def times(x: Int, y: Int) = x * y

    override def negate(x: Int) = -x

    override def fromInt(x: Int) = x

    override def toInt(x: Int) = x.toInt

    override def toLong(x: Int) = x.toLong

    override def toFloat(x: Int) = x.toFloat

    override def toDouble(x: Int) = x

    override def compare(x: Int, y: Int) = x.compareTo(y)
  }

}
