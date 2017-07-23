import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.math.Semiring
import org.scalatest.FlatSpec
import shapeless._
import shapeless.ops.nat.Sum

import scala.reflect.ClassTag

object Tensors {

  sealed trait Nat {
    type Add[A <: Nat] <: Nat // 1.add(5)
  }

  case class Zero() extends Nat {
    type Add[A <: Nat] = A
  }

  case class Succ[N <: Nat]() extends Nat {
    type Add[A <: Nat] = Succ[N#Add[A]]
  }

  type Plus[A <: Nat, B <: Nat] = A#Add[B]

  object Nat {
    type _0 = Zero
    type _1 = Succ[Zero]
    type _2 = Succ[_1]
    type _3 = Succ[_2]
  }

  case class Domain[K <: Nat](sizes: Seq[Int]) {
    lazy val size = sizes.product
  }

  object Domain {
    def apply(): Domain[Nat._0] = Domain(Seq(1))

    def apply(length: Int): Domain[Nat._1] = Domain(Seq(length))

    def join[K <: Nat, L <: Nat](domainK: Domain[K], domainL: Domain[L]) =
      Domain[Plus[K, L]](domainK.sizes ++ domainL.sizes)
  }


  class MyTensor[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](val dom: Domain[DOM], val mod: Domain[MOD], val data: DenseVector[V]) {

    def transpose: MyTensor[V, MOD, DOM] = MyTensor(mod, dom)
  }

  object MyTensor {

    def apply[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
      new MyTensor[V, DOM, MOD](dom, mod, DenseVector.zeros[V](dom.size * mod.size))

    def ones[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
      new MyTensor[V, DOM, MOD](dom, mod, DenseVector.ones[V](dom.size * mod.size))

    def eye[V: Semiring : ClassTag, DOM <: Nat](dom: Domain[DOM]) = {
      val matrix = DenseMatrix.eye[V](dom.size)
      new MyTensor[V, DOM, DOM](dom, dom, DenseVector(matrix.data))
    }

  }

  trait Expression[V, K <: Nat, CK <: Nat] {

    val dom: Domain[K]
    val mod: Domain[CK]

    def eval(): MyTensor[V, K, CK]

    def grad[L <: Nat](variable: Variable[V, L]): Expression[V, K, Plus[CK, L]]
  }

  class ConstantExpression[V: ClassTag : Semiring, K <: Nat, CK <: Nat]
  (
    val value: MyTensor[V, K, CK]
  ) extends Expression[V, K, CK] {

    val dom = value.dom
    val mod = value.mod

    override def eval() = value

    override def grad[L <: Nat](variable: Variable[V, L]) =
      new ConstantExpression(MyTensor(dom, Domain.join(mod, variable.dom)))
  }

  class Variable[V: Semiring, K <: Nat](val dom: Domain[K]) extends Expression[V, K, Nat._0] {

    val mod = Domain()

    override def eval() = {
      throw new NotImplementedError("Variable should not be evaluated")
    }

    def grad(variable: Variable[V, K]) = {
      if (variable == this) {
        new ConstantExpression[V, K, K](MyTensor.eye(dom))
      } else {
        new ConstantExpression[V, K, K](MyTensor(dom, dom))
      }
    }

    override def grad[L <: Nat](variable: Variable[V, L]) = {
      new ConstantExpression[V, K, L](MyTensor(dom, variable.dom))
    }
  }

  /*
  class VariableEmbedding[A, X](implicit eb: Embedding[A, X]) extends Embedding[Variable[A], Variable[X]] {
    override def embed(value: Variable[A]) = new Variable[X] {
      override def eval() = eb.embed(value.eval())

      override def grad[B, C](target: Variable[B])(implicit g: Gradient[X, B, C], rvc: Ring[Variable[C]], eb: Embedding[X, C]) = {

      }
    }
  }

  class VariableRing[A](implicit ringA: Ring[A]) extends Ring[Variable[A]] {

    override def neg(value: Variable[A]) = new Variable[A] {
      override def eval() = ringA.neg(value.eval())

      override def grad[B, C](target: Variable[B])(implicit g: Gradient[A, B, C], rvc: Ring[Variable[C]], eb: Embedding[A, C]) = {
        rvc.neg(value.grad(target))
      }
    }

    override def +(left: Variable[A], right: Variable[A]) = new Variable[A] {
      override def eval() = ringA.+(left.eval(), right.eval())

      override def grad[B, C](target: Variable[B])(implicit g: Gradient[A, B, C], rvc: Ring[Variable[C]], eb: Embedding[A, C]) = {
        val lg = left.grad(target)
        val rg = right.grad(target)
        rvc.+(lg, rg)
      }
    }

    override def -(left: Variable[A], right: Variable[A]) = new Variable[A] {
      override def eval() = ringA.-(left.eval(), right.eval())

      override def grad[B, C](target: Variable[B])(implicit g: Gradient[A, B, C], rvc: Ring[Variable[C]], eb: Embedding[A, C]) = {
        val lg = left.grad(target)
        val rg = right.grad(target)
        rvc.neg(lg, rg)
      }
    }

    override def *(left: Variable[A], right: Variable[A]) = new Variable[A] {
      override def eval() = ringA.*(left.eval(), right.eval())

      override def grad[B, C](target: Variable[B])(implicit g: Gradient[A, B, C], rvc: Ring[Variable[C]], eb: Embedding[A, C]) = {
        val lg = left.grad(target)
        val rg = right.grad(target)
        rvc.+(rvc.*(lg, right), rvc.*(left, rg))

      }
    }

    override def /(left: Variable[A], right: Variable[A]) = new Variable[A] {
      override def eval() = ringA./(left.eval(), right.eval())
    }
  }

  */

}

class TensorsSpec extends FlatSpec {

  import Tensors._

  "ed" should "have a reasonable type" in {
    val ed = Domain.join(Domain(2), Domain(3))
    println(ed.sizes)

    val a = Domain[Succ[Nat._0]](Seq(1))
    val b = Domain[Succ[Succ[Nat._0]]](Seq(1))
    val c: Domain[Nat._1] = a
    assert(ed == Domain[Nat._2](sizes = List(2, 3)))
  }
}
