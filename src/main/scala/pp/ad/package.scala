package pp

import scalaz._
import scalaz.Scalaz._

package object ad {

  trait ContainerOps[W[_]] {

    type Shape

    def shapeOf[V](value: W[V]): Shape

    def lift[A](value: A): W[A]

    def map[A, B](container: W[A])(fn: A => B): W[B]

    def zipMap[A, B, C](left: W[A], right: W[B])(fn: (A, B) => C): W[C]

    def foldLeft[A, B](container: W[A])(zero: B)(fn: (B, A) => B): B

    def eye[A](shape: Shape, one: A, zero: A): W[W[A]]

    def fill[A](shape: Shape, value: A): W[A]
  }

  object ContainerOps {

    type Aux[W[_], S] = ContainerOps[W] {type Shape = S}

    implicit val idOps = new ContainerOps[Id] {

      override type Shape =
        Any

      override def shapeOf[V](value: Scalaz.Id[V]) =
        null

      override def lift[A](value: A) =
        value

      override def map[A, B](container: Scalaz.Id[A])(fn: (A) => B) =
        fn(container)

      override def zipMap[A, B, C](left: Scalaz.Id[A], right: Scalaz.Id[B])(fn: (A, B) => C) =
        fn(left, right)

      override def foldLeft[A, B](container: A)(zero: B)(fn: (B, A) => B): B =
        fn(zero, container)

      override def eye[A](shape: Shape, one: A, zero: A) =
        one

      override def fill[A](shape: Shape, value: A) =
        value
    }

    implicit val listOps = new ContainerOps[List] {

      type Shape = Int

      override def shapeOf[V](value: List[V]) =
        value.size

      override def lift[A](value: A) = List(value)

      override def map[A, B](container: List[A])(fn: (A) => B) =
        container.map(fn)

      override def zipMap[A, B, C](left: List[A], right: List[B])(fn: (A, B) => C): List[C] = {
        val zipped = left.zip(right)
        zipped.map { x =>
          fn(x._1, x._2)
        }
      }

      override def foldLeft[A, B](container: List[A])(zero: B)(fn: (B, A) => B) =
        container.foldLeft(zero)(fn)

      override def eye[A](shape: Int, one: A, zero: A) = {
        Range(0, shape)
          .map { i =>
            Range(0, shape)
              .map { j =>
                if (i == j)
                  one
                else
                  zero
              }.toList
          }.toList
      }

      override def fill[A](shape: Int, value: A) = {
        Range(0, shape).map(_ => value).toList
      }
    }

  }

  type Gradient[W[_], A[_], V] = W[A[V]]

  trait Floating[V] extends Fractional[V] {

    def log: RichFunc[V]

    def sum: Accumulator[V]
  }

  object Floating {

    implicit val doubleFloating: Floating[Double] = new Floating[Double] {

      override def log = new RichFunc[Double] {

        def apply(x: Double) = scala.math.log(x)

        def grad(x: Double) = 1.0 / x
      }

      override def sum = new Accumulator[Double] {

        override def apply(a: Double, b: Double) = a + b

        override val start = 0.0

        override def grad(a: Double, b: Double) = 1.0
      }


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

    implicit def toFloating[U[_], V](implicit numV: Floating[V], ops: ContainerOps[U]): Floating[U[V]] = ValueType.toValueType[U, V]
  }

  trait ValueType[U[_], V] extends Floating[U[V]] {

    def valueVT: Floating[V]

  }

  object ValueType {

    implicit def toValueType[U[_], V](implicit numV: Floating[V], ops: ContainerOps[U]): ValueType[U, V] = new ValueType[U, V] {

      override def valueVT = numV

      override def log = new RichFunc[U[V]] {

        private val upstream = numV.log

        override def grad(v: U[V]) = ops.map(v)(upstream.grad)

        override def apply(v: U[V]) = ops.map(v)(upstream.apply)
      }

      override def sum = new Accumulator[U[V]] {

        private val upstream = numV.sum

        override def start = null.asInstanceOf[U[V]]

        override def apply(acc: U[V], v: U[V]) = ops.zipMap(acc, v) {
          (a, x) => upstream(if (a == null) upstream.start else a, x)
        }

        override def grad(a: U[V], v: U[V]) = ops.zipMap(a, v)(upstream)
      }

      override def div(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.div)

      override def plus(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.plus)

      override def minus(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.minus)

      override def times(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.times)

      override def negate(x: U[V]) = ops.map(x)(numV.negate)

      override def fromInt(x: Int) = ???

      override def toInt(x: U[V]) = ???

      override def toLong(x: U[V]) = ???

      override def toFloat(x: U[V]) = ???

      override def toDouble(x: U[V]) = ???

      override def compare(x: U[V], y: U[V]) = ???
    }
  }

  trait RichFunc[V] extends (V => V) {
    def grad(v: V): V
  }

  trait UnaryFunc {

    def apply[U[_], V](node: Node[U, V])(implicit vt: ValueType[U, V], ops: ContainerOps[U], impl: Impl[V]): Node[U, V] =
      UnaryNode(node, impl)

    trait Impl[V] extends RichFunc[V]

    def wrapFunc[V](fn: RichFunc[V]): Impl[V] = new Impl[V] {

      override def apply(v: V) = fn.apply(v)

      override def grad(v: V) = fn.grad(v)
    }
  }

  object log extends UnaryFunc {

    implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.log)
  }

  trait Accumulator[V] extends ((V, V) => V) {

    def start: V

    // gradient of the accumulator to v at a
    def grad(a: V, v: V): V
  }

  trait AccumulatingFunc {

    def apply[U[_], V](node: Node[U, V])(implicit vt: ValueType[U, V], idT: ValueType[Id, V], ops: ContainerOps[U], impl: Impl[V]): Node[Id, V] =
      AccumulatingNode(node, impl)

    trait Impl[V] extends Accumulator[V]

    def wrapFunc[V](fn: Accumulator[V]): Impl[V] = new Impl[V] {

      override def apply(acc: V, v: V) = fn.apply(acc, v)

      override def start = fn.start

      override def grad(acc: V, v: V) = fn.grad(acc, v)
    }
  }

  object sum extends AccumulatingFunc {

    implicit def impl[V](implicit numV: Floating[V]): Impl[V] = wrapFunc(numV.sum)
  }

  trait Node[U[_], V] extends (() => U[V]) {

    implicit val vt: ValueType[U, V]

    def grad[W[_] : ContainerOps](v: Variable[W, V]): Gradient[W, U, V]

    def unary_-() = Negate(this)

    def *(rhs: Node[U, V]) = Multiply(this, rhs)

    def /(rhs: Node[U, V]) = Divide(this, rhs)

    def +(rhs: Node[U, V]) = Add(this, rhs)

    def -(rhs: Node[U, V]) = Subtract(this, rhs)
  }

  case class UnaryNode[U[_] : ContainerOps, V](up: Node[U, V], rf: RichFunc[V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {
    private val ops = implicitly[ContainerOps[U]]

    override def apply() = {
      ops.map(up())(rf.apply)
    }

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val opsW = implicitly[ContainerOps[W]]
      val ug = up.grad(v)
      opsW.map(ug) { v =>
        vt.times(v, ops.map(up())(rf.grad))
      }
    }
  }

  case class AccumulatingNode[U[_] : ContainerOps, V, A](up: Node[U, V], rf: Accumulator[V])(implicit st: ValueType[U, V], val vt: ValueType[Id, V]) extends Node[Id, V] {
    private val ops = implicitly[ContainerOps[U]]

    override def apply() = {
      ops.foldLeft(up())(rf.start)(rf.apply)
    }

    // f(f(f(zero, x1), x2), x3)
    // grad_v =>
    //   (
    //    x3' +
    //    (
    //     x2' +
    //     (x1' * f'(zero, x1))
    //    ) * f'(f(zero, x1), x2) +
    //   ) * f'(f(f(zero, x1), x2), x3)

    // ug = (x1', x2', x3')

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val opsW = implicitly[ContainerOps[W]]
      val ug = up.grad(v)
      val result = opsW.map(ug) { g =>
        val vg = ops.zipMap(up(), g)((_, _))
        ops.foldLeft(vg)((rf.start, vt.zero)) {
          (acc, x) =>
            val (av, ag) = acc
            val (xv, xg) = x
            (
              rf(av, xv),
              vt.times(vt.plus(xg, ag), rf.grad(av, xv))
            )
        }
      }
      opsW.map(result)(_._2)
    }
  }

  trait ConstantLike[U[_], V] extends Node[U, V] {

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      val shape = ops.shapeOf(v())
      ops.fill(shape, vt.zero)
    }
  }

  case class Constant[U[_], V](value: U[V])(implicit val vt: ValueType[U, V]) extends ConstantLike[U, V] {
    override def apply() = value
  }

  case class Negate[U[_], V](up: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

    override def apply() = vt.negate(up())

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      ops.map(up.grad(v)) { g => vt.negate(g) }
    }

  }

  case class Multiply[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

    override def apply() = vt.times(lhs(), rhs())

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      val lv = lhs()
      val leftg = lhs.grad(v)
      val rv = rhs()
      val rightg = rhs.grad(v)
      ops.zipMap(leftg, rightg) {
        (lg, rg) =>
          vt.plus(
            vt.times(lg, rv),
            vt.times(lv, rg)
          )
      }
    }

  }


  case class Divide[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

    override def apply() = vt.div(lhs(), rhs())

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      val lv = lhs()
      val leftg = lhs.grad(v)
      val rv = rhs()
      val rightg = rhs.grad(v)
      ops.zipMap(leftg, rightg) {
        (lg, rg) =>
          vt.minus(
            vt.div(lg, rv),
            vt.div(
              vt.times(lv, rg),
              vt.times(rv, rv)
            )
          )
      }
    }

  }

  case class Add[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

    override def apply() = vt.plus(lhs(), rhs())

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      ops.zipMap(lhs.grad(v), rhs.grad(v)) {
        (lg, rg) => vt.plus(lg, rg)
      }
    }

  }

  case class Subtract[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

    override def apply() = vt.minus(lhs(), rhs())

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      ops.zipMap(lhs.grad(v), rhs.grad(v)) {
        (lg, rg) => vt.minus(lg, rg)
      }
    }

  }

  trait Model {

    def valueOf[U[_], V](v: Variable[U, V]): U[V]
  }

  trait Variable[U[_], V] extends Node[U, V] {

    override def grad[W[_] : ContainerOps](v: Variable[W, V]) = {
      val ops = implicitly[ContainerOps[W]]
      val shape = ops.shapeOf(v())
      if (this == v) {
        ops.eye(shape, vt.valueVT.one, vt.valueVT.zero).asInstanceOf[Gradient[W, U, V]]
      } else {
        ops.fill(shape, vt.zero)
      }
    }
  }

  object Variable {

    def apply[U[_], V](value: U[V])(implicit valueType: ValueType[U, V]): Variable[U, V] = new Variable[U, V] {
      override implicit val vt = valueType

      override def apply() = value
    }
  }

  trait Distribution[U[_], V] {

    def sample(implicit model: Model): Sample[U, V]

    def observe(data: U[V]): Observation[U, V]
  }

  trait Stochast[U[_], V] {
    def logp(): Node[Id, V]
  }

  trait Sample[U[_], V] extends Variable[U, V] with Stochast[U, V]

  trait Observation[U[_], V] extends Node[U, V] with Stochast[U, V]

  case class Normal[U[_] : ContainerOps, V](mu: Node[U, V], sigma: Node[U, V])(implicit vt: ValueType[U, V]) extends Distribution[U, V] {

    override def sample(implicit model: Model) = NormalSample(mu, sigma)

    override def observe(data: U[V]) = NormalObservation(mu, sigma, data)
  }

  trait NormalStochast[U[_], V] extends Stochast[U, V] {
    self: Node[U, V] =>

    def mu: Node[U, V]

    def sigma: Node[U, V]

    def vt: ValueType[U, V]

    implicit def ops: ContainerOps[U]

    override def logp(): Node[Id, V] = {
      implicit val numV = vt.valueVT
      val x = (this - mu) / sigma
      sum(-(log(sigma) - x * x / Constant(vt.fromInt(2))))
    }
  }

  case class NormalSample[U[_] : ContainerOps, V](mu: Node[U, V], sigma: Node[U, V])(implicit val vt: ValueType[U, V], model: Model)
    extends Sample[U, V] with NormalStochast[U, V] {

    override val ops = implicitly[ContainerOps[U]]

    override def apply() = model.valueOf(this)
  }

  case class NormalObservation[U[_] : ContainerOps, V](mu: Node[U, V], sigma: Node[U, V], value: U[V])(implicit val vt: ValueType[U, V])
    extends Observation[U, V] with NormalStochast[U, V] with ConstantLike[U, V] {

    override val ops = implicitly[ContainerOps[U]]

    override def apply() = value
  }

}
