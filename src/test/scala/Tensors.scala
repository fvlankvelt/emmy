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
  sealed trait Node
  case class Leaf() extends Node
  case class Fork[L <: Node, R <: Node](l: L, r: R) extends Node

  val tree = Fork(Fork(Leaf(), Leaf()), Leaf())
  */

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

    //    implicit def assoc[N <: Nat, M <: Nat, P <: Nat](implicit ih: PlusAssoc[N, M, P]) = ih

  }

  trait TensorType {
    type K <: Nat
    type CK <: Nat
  }

  type TT[L <: Nat, CL <: Nat] = TensorType {
    type K = L
    type CK = CL
  }

  trait Gradient[V, T <: TensorType, E, M <: Nat] {
    def grad(expression: E, variable: Variable[V, TT[M, Nat._0]]): Expression[V, TT[T#K, Plus[T#CK, M]]]
  }

  object Gradient {

    implicit def constantToGradient[V: Semiring : ClassTag, T <: TensorType, M <: Nat]: Gradient[V, T, ConstantExpression[V, T], M] =
      new Gradient[V, T, ConstantExpression[V, T], M] {
        override def grad(expression: ConstantExpression[V, T], variable: Variable[V, TT[M, Nat._0]]) = {
          val dom = expression.dom
          val mod = expression.mod
          new ConstantExpression[V, TT[T#K, Plus[T#CK, M]]](MyTensor[V, T#K, Plus[T#CK, M]](dom, Domain.join(mod, variable.dom)))
        }
      }

    implicit def variableGradient[V: Semiring : ClassTag, T <: TensorType, M <: Nat]: Gradient[V, T, Variable[V, T], M] =
      new Gradient[V, T, Variable[V, T], M] {
        override def grad(expression: Variable[V, T], variable: Variable[V, TT[M, Nat._0]]): Expression[V, TT[T#K, T#CK#Add[M]]] = {
          type CR = T#CK#Add[M]
          val dom = expression.dom
          if (expression eq variable) {
            new ConstantExpression[V, TT[T#K, CR]](MyTensor.eye(dom).asInstanceOf[MyTensor[V, T#K, CR]]) // ugh, but cast will succeed
          } else {
            new ConstantExpression[V, TT[T#K, CR]](MyTensor[V, T#K, CR](dom, Domain.join(expression.mod, variable.dom)))
          }
        }
      }

    implicit def outerGradient[V: Semiring : ClassTag, TL <: TensorType, TR <: TensorType, M <: Nat](implicit assoc: PlusAssoc[TR#CK, TL#CK, M]): Gradient[V, TT[Plus[TL#K, TR#K], Plus[TR#CK, TL#CK]], OuterExpression[V, TL, TR], M] =
      new Gradient[V, TT[Plus[TL#K, TR#K], Plus[TR#CK, TL#CK]], OuterExpression[V, TL, TR], M] {
        override def grad(expr: OuterExpression[V, TL, TR], variable: Variable[V, TT[M, Nat._0]]) = {
//          implicit val assoc = implicitly[PlusAssoc[TL#CK, TR#CK, M]].proof
          val s = assoc.proof.subst[({type N[X] = Expression[V, TT[Plus[TL#K, TR#K], X]]})#N] _

          val leftGrad : Expression[V, TT[TL#K, Plus[TL#CK, M]]] = expr.left match {
            case c @ ConstantExpression(_) =>
              val gradient = constantToGradient[V, TL, M]
              gradient.grad(c, variable)
            case v @ Variable(_, _) =>
              val gradient = variableGradient[V, TL, M]
              gradient.grad(v, variable) // ugh, but should work
//            case o @ OuterExpression(_, _) =>
//              val gradient = outerGradient[V, o.L, o.R]
//              gradient.grad(o.asInstanceOf[OuterExpression[V, o.L, o.R]], variable)
          }

//          val leftGrad = functions.grad(expr.left, variable)
          val leftRes = s(leftGrad outer expr.right)
          leftRes
        }
      }
  }

  object functions {
    def grad[V, K <: Nat, CK <: Nat, M <: Nat, E]
    (
      expression: E, variable: Variable[V, TT[M, Nat._0]]
    )(
      implicit gradient: Gradient[V, TT[K, CK], E, M]
    ): Expression[V, TT[K, Plus[CK, M]]] = {
      gradient.grad(expression, variable)
    }
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


  sealed trait Expression[V, T <: TensorType] {

    implicit val ringV: Semiring[V]
    implicit val ctV: ClassTag[V]

    //    type AddGrad[P <: Nat] <: Nat// = PlusAssoc[CK, Zero, P]

    val dom: Domain[T#K]
    val mod: Domain[T#CK]

    def eval(): MyTensor[V, T#K, T#CK]

    //    def grad[L <: Nat : AddGrad](variable: Variable[V, L]): Expression[V, K, Plus[CK, L]]

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
    def shiftLeft: Expression[V, TT[Succ[T#K], T#CK#P]] = {
      val self = this
      new Expression[V, TT[Succ[T#K], T#CK#P]] {
        implicit val ringV = self.ringV
        implicit val ctV = self.ctV
        override val dom = Domain[Succ[T#K]](self.dom.sizes :+ self.mod.sizes.head)
        override val mod = Domain[T#CK#P](self.mod.sizes.drop(1))

        override def eval() = {
          val tensor = self.eval()
          val data = tensor.data
          val reshaped = data.reshape(dom.size, mod.size)
          new MyTensor[V, Succ[T#K], T#CK#P](dom, mod, reshaped)
        }

        //        override def grad[L <: Nat : AddGrad](variable: Variable[V, L]) = ???
      }
    }

    def transpose: Expression[V, TT[T#CK, T#K]] = {
      val self = this
      new Expression[V, TT[T#CK, T#K]] {
        implicit val ringV = self.ringV
        implicit val ctV = self.ctV
        override val dom = self.mod
        override val mod = self.dom

        override def eval() = self.eval().transpose

        //        override def grad[L <: Nat : AddGrad](variable: Variable[V, L]) = ???
      }
    }

    /*
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

        //        override def grad[L <: Nat : AddGrad](variable: Variable[V, L]) = ???
      }

    }
    */

    def outer[OT <: TensorType](other: Expression[V, OT]): Expression[V, TT[Plus[T#K, OT#K], Plus[OT#CK, T#CK]]] =
      new OuterExpression[V, T, OT](this, other)

  }

  case class OuterExpression[V: ClassTag : Semiring, TL <: TensorType, TR <: TensorType]
  (
    left: Expression[V, TL],
    right: Expression[V, TR]
  ) extends Expression[V, TT[Plus[TL#K, TR#K], Plus[TR#CK, TL#CK]]] {
    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val dom = Domain.join(left.dom, right.dom)
    val mod = Domain.join(right.mod, left.mod)

    override def eval() = {
      val leftMatrix = left.eval().data
      val rightMatrix = right.eval().data
      val newMatrix = DenseMatrix.zeros[V](dom.size, mod.size)
      for {
        rowLeft <- 0 until left.dom.size
        rowRight <- 0 until right.dom.size
        colLeft <- 0 until left.mod.size
        colRight <- 0 until right.mod.size
      } {
        val entry = ringV.*(
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

    //        override type AddGrad[M <: Nat] = PlusAssoc[CL, CK, M]

  }

  case class ConstantExpression[V: ClassTag : Semiring, T <: TensorType]
  (
    value: MyTensor[V, T#K, T#CK]
  ) extends Expression[V, T] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val dom = value.dom
    val mod = value.mod

    override def eval() = value

  }

  case class Variable[V: Semiring : ClassTag, T <: TensorType] private (
    dom: Domain[T#K],
    mod: Domain[T#CK]
  ) extends Expression[V, T] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]


    override def eval() = {
      throw new NotImplementedError("Variable should not be evaluated")
    }

  }

  object Variable {
    def apply[V: Semiring : ClassTag, L <: Nat](dom: Domain[L]) =
      Variable[V, TT[L, Nat._0]](dom, Domain())
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
    import Gradient._
    import functions._
    val a = Variable[Float, Nat._0](Domain())
    val b = Variable[Float, Nat._1](Domain(2))
    val c = grad(a, a)
    val d = grad(a, b)
  }

  trait TestAssoc[A <: Nat, B <: Nat, C <: Nat] {
    def assoc: PlusAssoc[A, B, C]

    def proof = assoc.proof
  }

  trait TestPartial[A <: Nat, B <: Nat] {
    type Assoc[C <: Nat] = PlusAssoc[A, B, C]

    def toAssoc[C <: Nat : Assoc]() = {
      implicitly[Assoc[C]].proof
    }
  }

  "addition" should "be associative" in {
    type A = Nat._1
    type B = Nat._2
    type C = Nat._3
    import PlusAssoc._
    val proof = implicitly[PlusAssoc[A, B, C]].proof
    //    PlusAssoc.assoc[A, B, C].proof

    val assocType = implicitly[PlusAssoc[A, B, C]]
    val testAssoc = new TestAssoc[A, B, C] {
      override val assoc = assocType
    }

    val testPartial = new TestPartial[A, B] {}
    testPartial.toAssoc[C]()
    //    val a: Plus[A, Plus[B, C]] = null
    //    val b: Plus[Plus[A, B], C] = a
    //    val c: Plus[A, Plus[B, C]] = b
  }

}
