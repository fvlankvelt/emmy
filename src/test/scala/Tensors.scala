import breeze.linalg.DenseMatrix
import breeze.math.Semiring
import org.scalatest.FlatSpec

import scala.language.higherKinds
import scala.reflect.ClassTag
import scalaz.Leibniz

object Tensors {

  sealed trait Nat {
    type P <: Nat
    type Add[A <: Nat] <: Nat // 1.add(5)
  }

  object Nat {
    // Equality on nats
    type ===[A <: Nat, B <: Nat] = Leibniz[Nothing, Nat, A, B]

    type _0 = Zero
    type _1 = Succ[Zero]
    type _2 = Succ[_1]
    type _3 = Succ[_2]
    type _4 = Succ[_3]
    type _5 = Succ[_4]
    type _6 = Succ[_5]
  }

  case class Zero() extends Nat {
    type P = Zero
    type Add[A <: Nat] = A
  }

  case class Succ[N <: Nat]() extends Nat {
    type P = N
    type Add[A <: Nat] = Succ[N#Add[A]]
  }

  type Plus[A <: Nat, B <: Nat] = A#Add[B]

  /*
  plus-assoc : ∀ n m p → (n + (m + p)) ≡ ((n + m) + p)
  plus-assoc zero m p = refl
  plus-assoc (suc n) m p = cong suc (plus-assoc n m p)
*/
  trait PlusAssoc[N <: Nat, M <: Nat, P <: Nat] {
    import Tensors.Nat.===

    val proof: Plus[N, Plus[M, P]] === Plus[Plus[N, M], P]
  }

  object PlusAssoc {
    import Tensors.Nat.===

    implicit def plusAssocZero[N <: Nat, M <: Nat]: PlusAssoc[Zero, N, M] = new PlusAssoc[Zero, N, M] {
      val proof: Plus[N, M] === Plus[N, M] = Leibniz.refl
    }

    implicit def plusAssocSucc[N <: Nat, M <: Nat, P <: Nat](implicit
                                                             ih: PlusAssoc[N, M, P]): PlusAssoc[Succ[N], M, P] = new PlusAssoc[Succ[N], M, P] {
      // For some reason scalac fails to infer right params for lift :(
      val proof: Succ[Plus[N, Plus[M, P]]] === Succ[Plus[Plus[N, M], P]] = Leibniz.lift[
        Nothing, Nothing,
        Nat, Nat,
        Succ,
        Plus[N, Plus[M, P]], Plus[Plus[N, M], P]
        ](ih.proof)
    }

    implicit def assoc[N <: Nat, M <: Nat, P <: Nat](implicit ih: PlusAssoc[N, M, P]) = ih

  }

  /*
  case class PlusMove[N <: Nat, M <: Nat]() {

    type assoc[P <: Nat] = PlusAssoc[N, M, P]
  }

  case class PlusPreMove[N <: Nat]() {
    type prep[M <: Nat] = PlusMove[N, M]
  }

  object PlusPreMove {

    implicit def plusPreMoveZero: PlusPreMove[Zero] = PlusPreMove[Zero]()

    implicit def plusPreMoveNext[N <: Nat](implicit prev: PlusPreMove[N]): PlusPreMove[Succ[N]] =
      new PlusPreMove[Succ[N]] {}
  }
  */

  case class Domain[K <: Nat](sizes: Seq[Int]) {
    lazy val size = sizes.product
  }

  object Domain {
    def apply(): Domain[Nat._0] = Domain(Seq(1))

    def apply(length: Int): Domain[Nat._1] = Domain(Seq(length))

    def join[K <: Nat, L <: Nat](domainK: Domain[K], domainL: Domain[L]) =
      Domain[Plus[K, L]](domainK.sizes ++ domainL.sizes)
  }


  class MyTensor[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](val dom: Domain[DOM], val mod: Domain[MOD], val data: DenseMatrix[V]) {

    def transpose: MyTensor[V, MOD, DOM] = new MyTensor(mod, dom, data.t)
  }

  object MyTensor {

    def apply[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
      new MyTensor[V, DOM, MOD](dom, mod, DenseMatrix.zeros[V](dom.size, mod.size))

    def ones[V: Semiring : ClassTag, DOM <: Nat, MOD <: Nat](dom: Domain[DOM], mod: Domain[MOD]) =
      new MyTensor[V, DOM, MOD](dom, mod, DenseMatrix.ones[V](dom.size, mod.size))

    def eye[V: Semiring : ClassTag, DOM <: Nat](dom: Domain[DOM]) = {
      new MyTensor[V, DOM, DOM](dom, dom, DenseMatrix.eye[V](dom.size))
    }

  }


  trait Expression[V, K <: Nat, CK <: Nat] {

    implicit val ringV: Semiring[V]
    implicit val ctV: ClassTag[V]

    val dom: Domain[K]
    val mod: Domain[CK]

    def eval(): MyTensor[V, K, CK]

    def grad[L <: Nat](variable: Variable[V, L]): Expression[V, K, Plus[CK, L]]

    // ex:   dom = (3)     mod = (2, 3)
    //  =>   dom = (3, 2)  mod = (3)
    //
    // ORIG:
    // col = m_1 + m_2 * 2
    // row = d_1
    // idx = d_1 + 3 * (m_1 + 2 * m_2) = d_1 + 3 * m_1 + 6 * m_2
    //
    // NEW:
    // col = m_2
    // row = d_1 + 3 * m_1
    // idx = d_1 + 3 * m_1 + 6 * m_2
    def shiftLeft: Expression[V, Succ[K], CK#P] = {
      val self = this
      new Expression[V, Succ[K], CK#P] {
        implicit val ringV = self.ringV
        implicit val ctV = self.ctV
        override val dom = Domain[Succ[K]](self.dom.sizes :+ self.mod.sizes.head)
        override val mod = Domain[CK#P](self.mod.sizes.drop(1))

        override def eval() = {
          val tensor = self.eval()
          val data = tensor.data
          val reshaped = data.reshape(dom.size, mod.size)
          new MyTensor[V, Succ[K], CK#P](dom, mod, reshaped)
        }

        override def grad[L <: Nat](variable: Variable[V, L]) = ???
      }
    }

    def transpose: Expression[V, CK, K] = {
      val self = this
      new Expression[V, CK, K] {
        implicit val ringV = self.ringV
        implicit val ctV = self.ctV
        override val dom = self.mod
        override val mod = self.dom

        override def eval() = self.eval().transpose

        override def grad[L <: Nat](variable: Variable[V, L]) = ???
      }
    }

    def trace: Expression[V, K#P, CK#P] = {
      assert(dom.sizes.last == mod.sizes.head)
      val self = this
      new Expression[V, K#P, CK#P] {
        implicit val ringV = self.ringV
        implicit val ctV = self.ctV
        override val dom = Domain[K#P](self.dom.sizes.dropRight(1))
        override val mod = Domain[CK#P](self.mod.sizes.drop(1))

        override def eval() = {
          val matrix = self.eval().data
          val newMatrix = DenseMatrix.zeros[V](dom.size, mod.size)
          for {
            row <- 0 until dom.size
            col <- 0 until mod.size
          } {
            val length = self.dom.sizes.head
            val ring = implicitly[Semiring[V]]
            var sum = ring.zero
            for {i <- 0 until length} {
              sum = ring.+(sum, matrix(row * length + i, col * length + i))
            }
            newMatrix(row, col) = sum
          }
          new MyTensor(dom, mod, newMatrix)
        }

        override def grad[L <: Nat](variable: Variable[V, L]) = ???
      }

    }

    def outer[L <: Nat, CL <: Nat](other: Expression[V, L, CL]): Expression[V, Plus[K, L], Plus[CL, CK]] = {
      val left = this
      val right = other
      new Expression[V, Plus[K, L], Plus[CL, CK]] {
        implicit val ringV = left.ringV
        implicit val ctV = left.ctV
        val dom = Domain.join(left.dom, right.dom)
        val mod = Domain.join(right.mod, left.mod)

        override def eval() = {
          val ring = implicitly[Semiring[V]]

          val leftMatrix = left.eval().data
          val rightMatrix = right.eval().data
          val newMatrix = DenseMatrix.zeros[V](dom.size, mod.size)
          for {
            rowLeft <- 0 until left.dom.size
            rowRight <- 0 until right.dom.size
            colLeft <- 0 until left.mod.size
            colRight <- 0 until right.mod.size
          } {
            val entry = ring.*(
              leftMatrix(rowLeft, colLeft),
              rightMatrix(rowRight, colRight)
            )
            newMatrix(
              rowLeft * right.dom.size + rowRight,
              colLeft * right.mod.size + colRight
            ) = entry
          }
          new MyTensor(dom, mod, newMatrix)
        }

        override def grad[M <: Nat](variable: Variable[V, M]) = ??? /* {
          implicit val assoc = PlusAssoc.assoc[CL, CK, M].proof
          val s = assoc.subst[({type N[X] = Expression[V, Plus[K, L], X]})#N]

          val leftGrad = left.grad(variable)
          val leftRes = s(leftGrad outer right)
          leftRes
        }
         */
      }
    }
  }

  class ConstantExpression[V: ClassTag : Semiring, K <: Nat, CK <: Nat]
  (
    val value: MyTensor[V, K, CK]
  ) extends Expression[V, K, CK] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val dom = value.dom
    val mod = value.mod

    override def eval() = value

    override def grad[L <: Nat](variable: Variable[V, L]) =
      new ConstantExpression[V, K, Plus[CK, L]](MyTensor[V, K, Plus[CK, L]](dom, Domain.join(mod, variable.dom)))
  }

  class Variable[V: Semiring : ClassTag, K <: Nat](val dom: Domain[K]) extends Expression[V, K, Nat._0] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val mod = Domain()

    override def eval() = {
      throw new NotImplementedError("Variable should not be evaluated")
    }

    override def grad[L <: Nat](variable: Variable[V, L]) = {
      if (variable eq this) {
        new ConstantExpression(MyTensor.eye(dom).asInstanceOf[MyTensor[V, K, L]]) // ugh, but cast will succeed
      } else {
        new ConstantExpression(MyTensor[V, K, L](dom, variable.dom))
      }
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

  "variable" should "have covariant gradient" in {
    val a = new Variable[Float, Nat._0](Domain())
    val b = new Variable[Float, Nat._1](Domain(2))
    val c = a.grad(a)
    val d = a.grad(b)
  }

  "addition" should "be associative" in {
    type A = Nat._1
    type B = Nat._2
    type C = Nat._3
    import PlusAssoc._
    val proof = implicitly[PlusAssoc[A, B, C]].proof
//    val a: Plus[A, Plus[B, C]] = null
//    val b: Plus[Plus[A, B], C] = a
//    val c: Plus[A, Plus[B, C]] = b
  }

}
