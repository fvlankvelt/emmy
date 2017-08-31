package pp

import pp.ad.{ContainerOps, Gradient}

import scalaz.{Scalaz, _}
import scalaz.Scalaz._
import scala.math.Numeric

package object ad {

  trait ContainerOps[W[_]] {

    type Shape

    def shapeOf[V](value: W[V]): Shape

    def headOption[V](value: W[V]): Option[V]

    def map[A, B](container: W[A])(fn: A => B): W[B]

    def zipMap[A, B, C](left: W[A], right: W[B])(fn: (A, B) => C): W[C]

    def eye[A](shape: Shape, one: A, zero: A): W[W[A]]

    def fill[A](shape: Shape, value: A): W[A]
  }

  object ContainerOps {

    type Aux[W[_], S] = ContainerOps[W] {type Shape = S}

    implicit val idOps = new ContainerOps[Id] {

      override type Shape = Any

      override def shapeOf[V](value: Scalaz.Id[V]) = null

      override def headOption[V](value: Scalaz.Id[V]) = Some(value)

      override def map[A, B](container: Scalaz.Id[A])(fn: (A) => B) = fn(container)

      override def zipMap[A, B, C](left: Scalaz.Id[A], right: Scalaz.Id[B])(fn: (A, B) => C) = fn(left, right)

      override def eye[A](shape: Shape, one: A, zero: A) = one

      override def fill[A](shape: Shape, value: A) = value
    }

    implicit val listOps = new ContainerOps[List] {
      type Shape = Int

      override def shapeOf[V](value: List[V]) = value.size

      override def headOption[V](value: List[V]) = value.headOption

      override def map[A, B](container: List[A])(fn: (A) => B) = container.map(fn)

      override def zipMap[A, B, C](left: List[A], right: List[B])(fn: (A, B) => C): List[C] = {
        val zipped = left.zip(right)
        zipped.map { x =>
          fn(x._1, x._2)
        }
      }

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

    /*
    implicit def composeOps[X[_], Y[_], XS, YS](implicit opsA: Aux[X, XS], opsB: Aux[Y, YS]): ContainerOps[({type Z[A] = X[Y[A]]})#Z] =
      new ContainerOps[({type Z[A] = X[Y[A]]})#Z] {

        override type Shape = (XS, YS)

        override def shapeOf[V](value: X[Y[V]]) =
          (opsA.shapeOf(value), opsB.shapeOf(opsA.headOption(value).get))

        override def headOption[V](value: X[Y[V]]) =
          opsA.headOption(value).flatMap(opsB.headOption)

        override def map[A, B](container: X[Y[A]])(fn: (A) => B) = {
          opsA.map(container) { y =>
            opsB.map(y) { a => fn(a) }
          }
        }

        override def zipMap[A, B, C](left: X[Y[A]], right: X[Y[B]])(fn: (A, B) => C) = {
          opsA.zipMap(left, right) { (l, r) =>
            opsB.zipMap(l, r) { (lv, rv) => fn(lv, rv) }
          }
        }

        override def eye[A](shape: Shape, one: A, zero: A): X[Y[X[Y[A]]]] = ???

        override def fill[A](shape: Shape, value: A) = opsA.fill(shape._1, opsB.fill(shape._2, value))
      }
      */
  }

  type Gradient[W[_], A[_], V] = W[A[V]]

}

trait ValueType[U[_], V] extends Numeric[U[V]] {

  def valueVT: Numeric[V]
}

object ValueType {

  implicit def idType[V](implicit numV: Numeric[V]): ValueType[Id, V] = new ValueType[Id, V] {

    val valueVT = numV

    override def plus(x: Scalaz.Id[V], y: Scalaz.Id[V]) = numV.plus(x, y)

    override def minus(x: Scalaz.Id[V], y: Scalaz.Id[V]) = numV.minus(x, y)

    override def times(x: Scalaz.Id[V], y: Scalaz.Id[V]) = numV.times(x, y)

    override def negate(x: Scalaz.Id[V]) = numV.negate(x)

    override def fromInt(x: Int) = numV.fromInt(x)

    override def toInt(x: Scalaz.Id[V]) = numV.toInt(x)

    override def toLong(x: Scalaz.Id[V]) = numV.toLong(x)

    override def toFloat(x: Scalaz.Id[V]) = numV.toFloat(x)

    override def toDouble(x: Scalaz.Id[V]) = numV.toDouble(x)

    override def compare(x: Scalaz.Id[V], y: Scalaz.Id[V]) = numV.compare(x, y)
  }

  implicit def listType[V](implicit numV: Numeric[V]): ValueType[List, V] = new ValueType[List, V] {

    val valueVT = numV

    override def plus(x: List[V], y: List[V]) = (x zip y).map { case (xv, yv) => numV.plus(xv, yv) }

    override def minus(x: List[V], y: List[V]) = (x zip y).map { case (xv, yv) => numV.minus(xv, yv) }

    override def times(x: List[V], y: List[V]) = (x zip y).map {
      case (xv, yv) =>
        numV.times(xv, yv)
    }

    override def negate(x: List[V]) = x.map(numV.negate)

    override def toInt(x: List[V]) = ???

    override def toLong(x: List[V]) = ???

    override def toFloat(x: List[V]) = ???

    override def toDouble(x: List[V]) = ???

    override def compare(x: List[V], y: List[V]) = ???

    override def fromInt(x: Int) = List(numV.fromInt(x))
  }
}

trait Node[U[_], V] extends (() => U[V]) {

  implicit val vt: ValueType[U, V]

  def grad[W[_] : ContainerOps](v: Var[W, V]): Gradient[W, U, V]

  def unary_-() = Negate(this)

  def *(rhs: Node[U, V]) = Multiply(this, rhs)

  def +(rhs: Node[U, V]) = Add(this, rhs)

  def -(rhs: Node[U, V]) = Subtract(this, rhs)
}

trait RichFunc[V] extends (V => V) {
  def grad(v: V): V
}

trait UnaryFunc {

  def apply[U[_], V](node: Node[U, V])(implicit vt: ValueType[U, V], ops: ContainerOps[U], impl: Impl[V]): Node[U, V] =
    UnaryNode(node, impl)

  trait Impl[V] extends RichFunc[V]

}

object log extends UnaryFunc {

  implicit val logDouble = new Impl[Double] {

    def apply(x: Double) = scala.math.log(x)

    def grad(x: Double) = 1.0 / x
  }
}

case class UnaryNode[U[_] : ContainerOps, V](up: Node[U, V], rf: RichFunc[V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {
  private val ops = implicitly[ContainerOps[U]]

  override def apply() = {
    ops.map(up())(rf.apply)
  }

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = up.grad(v)
    opsW.map(ug) { v =>
      vt.times(v, ops.map(up())(rf.grad))
    }
  }
}

case class Var[U[_], V](data: U[V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {
  private val self = this

  override def apply() = data

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
    val ops = implicitly[ContainerOps[W]]
    val shape = ops.shapeOf(v())
    if (self == v) {
      ops.eye(shape, vt.valueVT.one, vt.valueVT.zero).asInstanceOf[Gradient[W, U, V]]
    } else {
      ops.fill(shape, vt.zero)
    }
  }
}

case class Negate[U[_], V](up: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

  override def apply() = vt.negate(up())

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
    val ops = implicitly[ContainerOps[W]]
    ops.map(up.grad(v)) { g => vt.negate(g) }
  }

}

case class Multiply[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

  override def apply() = vt.times(lhs(), rhs())

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
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

case class Add[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

  override def apply() = vt.plus(lhs(), rhs())

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
    val ops = implicitly[ContainerOps[W]]
    ops.zipMap(lhs.grad(v), rhs.grad(v)) {
      (lg, rg) => vt.plus(lg, rg)
    }
  }

}

case class Subtract[U[_], V](lhs: Node[U, V], rhs: Node[U, V])(implicit val vt: ValueType[U, V]) extends Node[U, V] {

  override def apply() = vt.minus(lhs(), rhs())

  override def grad[W[_] : ContainerOps](v: Var[W, V]) = {
    val ops = implicitly[ContainerOps[W]]
    ops.zipMap(lhs.grad(v), rhs.grad(v)) {
      (lg, rg) => vt.minus(lg, rg)
    }
  }

}
