import breeze.linalg.{DenseMatrix, View}
import breeze.math.Semiring
import org.scalatest.FlatSpec

import scala.language.higherKinds
import scala.reflect.ClassTag

object Tensors {

  sealed trait Nat {
    type P <: Nat
    type Add[A <: Nat] <: Nat // 1.add(5)
    type Sub[A <: Nat] <: Nat // 5.sub(1)
  }

  object Nat {
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
    type Sub[A <: Nat] = A
  }

  case class Succ[N <: Nat]() extends Nat {
    type P = N
    type Add[A <: Nat] = Succ[N#Add[A]]
    type Sub[A <: Nat] = Succ[N#Sub[A#P]]
  }

  type Plus[A <: Nat, B <: Nat] = A#Add[B]
  type Min[A <: Nat, B <: Nat] = A#Sub[B]

  trait ToInt[N <: Nat] {
    def apply(): Int
  }

  object ToInt {
    implicit val zeroInt: ToInt[Zero] = new ToInt[Zero] {
      override def apply() = 0
    }

    implicit def succToInt[N <: Nat](implicit prev: ToInt[N]): ToInt[Succ[N]] = new ToInt[Succ[N]] {
      override def apply() = prev.apply() + 1
    }
  }

  case class Domain[K <: Nat : ToInt](sizes: Seq[Int]) {
    lazy val size = sizes.product
    implicit val toInt = implicitly[ToInt[K]]
  }

  object Domain {
    def apply(): Domain[Nat._0] = Domain(Seq.empty)

    def apply(length: Int): Domain[Nat._1] = Domain(Seq(length))

    def join[K <: Nat, L <: Nat](domainK: Domain[K], domainL: Domain[L]): Domain[Plus[K, L]] = {
      implicit val outInt = new ToInt[Plus[K, L]] {
        def apply() = domainK.toInt.apply() + domainL.toInt.apply()
      }
      Domain[Plus[K, L]](domainK.sizes ++ domainL.sizes)
    }
  }

  case class TensorShape[K <: Nat, CK <: Nat](dom: Domain[K], mod: Domain[CK]) {

    private implicit val intK = dom.toInt
    private implicit val intCK = mod.toInt

    def shiftLeft[L <: Nat : ToInt]: TensorShape[Plus[K, L], Min[CK, L]] = {
      val self = this
      val m = implicitly[ToInt[L]].apply()
      implicit val plusInt = new ToInt[Plus[K, L]] {
        override def apply() = implicitly[ToInt[K]].apply() + implicitly[ToInt[L]].apply()
      }
      implicit val minInt = new ToInt[Min[CK, L]] {
        override def apply() = implicitly[ToInt[CK]].apply() - implicitly[ToInt[L]].apply()
      }
      TensorShape(
        Domain[Plus[K, L]](self.dom.sizes ++ self.mod.sizes.take(m)),
        Domain[Min[CK, L]](self.mod.sizes.drop(m))
      )
    }

    def shiftRight[L <: Nat : ToInt]: TensorShape[Min[K, L], Plus[CK, L]] = {
      val self = this
      val m = implicitly[ToInt[L]].apply()
      implicit val plusInt = new ToInt[Min[K, L]] {
        override def apply() = implicitly[ToInt[K]].apply() - implicitly[ToInt[L]].apply()
      }
      implicit val minInt = new ToInt[Plus[CK, L]] {
        override def apply() = implicitly[ToInt[CK]].apply() + implicitly[ToInt[L]].apply()
      }
      TensorShape(
        Domain[Min[K, L]](self.dom.sizes.dropRight(m)),
        Domain[Plus[CK, L]](self.dom.sizes.takeRight(m) ++ self.mod.sizes)
      )
    }

    def transpose[L <: Nat : ToInt, CL <: Nat : ToInt]: TensorShape[Plus[Min[K, L], CL], Plus[Min[CK, CL], L]] = {
      val self = this
      val l = implicitly[ToInt[L]].apply()
      val cl = implicitly[ToInt[CL]].apply()
      implicit val varInt = new ToInt[Plus[Min[K, L], CL]] {
        override def apply() = implicitly[ToInt[K]].apply() - implicitly[ToInt[L]].apply() + implicitly[ToInt[CL]].apply()
      }
      implicit val covInt = new ToInt[Plus[Min[CK, CL], L]] {
        override def apply() = implicitly[ToInt[CK]].apply() - implicitly[ToInt[CL]].apply() + implicitly[ToInt[L]].apply()
      }
      TensorShape(
        Domain[Plus[Min[K, L], CL]](self.dom.sizes.dropRight(l) ++ self.mod.sizes.take(cl)),
        Domain[Plus[Min[CK, CL], L]](self.dom.sizes.takeRight(l) ++ self.mod.sizes.drop(cl))
      )
    }

    def outer[L <: Nat, CL <: Nat](right: TensorShape[L, CL]): TensorShape[Plus[K, L], Plus[CL, CK]] = {
      val left = this
      implicit val varInt = new ToInt[Plus[K, L]] {
        override def apply() = left.dom.toInt.apply() + right.dom.toInt.apply()
      }
      implicit val covInt = new ToInt[Plus[CL, CK]] {
        override def apply() = left.mod.toInt.apply() + right.mod.toInt.apply()
      }
      TensorShape(
        Domain.join(left.dom, right.dom),
        Domain.join(right.mod, left.mod)
      )
    }

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


  sealed trait Expression[V, K <: Nat, CK <: Nat] {

    implicit val ringV: Semiring[V]
    implicit val ctV: ClassTag[V]

    def tt: TensorShape[K, CK]

    def eval(): MyTensor[V, K, CK]

    def grad[M <: Nat : ToInt](variable: Variable[V, M]): Expression[V, K, Plus[M, CK]]

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
    def shiftLeft[M <: Nat : ToInt]: Expression[V, Plus[K, M], Min[CK, M]] = ShiftLeftExpression(this)

    def shiftRight[M <: Nat : ToInt]: Expression[V, Min[K, M], Plus[CK, M]] = ShiftRightExpression(this)

    def transpose[L <: Nat : ToInt, CL <: Nat : ToInt]: Expression[V, Plus[Min[K, L], CL], Plus[Min[CK, CL], L]] =
      TransposeExpression[V, K, CK, L, CL](this)

    def +(other: Expression[V, K, CK]): Expression[V, K, CK] = PlusExpression(this, other)

    def outer[OK <: Nat : ToInt, OCK <: Nat : ToInt](other: Expression[V, OK, OCK]): Expression[V, Plus[K, OK], Plus[OCK, CK]] =
      new OuterExpression[V, K, CK, OK, OCK](this, other)
  }

  case class ShiftLeftExpression[
  V: ClassTag : Semiring,
  K <: Nat,
  CK <: Nat,
  L <: Nat : ToInt
  ](self: Expression[V, K, CK]) extends Expression[V, Plus[K, L], Min[CK, L]] {

    val ringV = self.ringV
    val ctV = self.ctV

    override val tt = self.tt.shiftLeft[L]

    override def eval() = {
      val tensor = self.eval()
      val data = tensor.data
      val reshaped = data.reshape(tt.dom.size, tt.mod.size, View.Require)
      new MyTensor[V, Plus[K, L], Min[CK, L]](tt.dom, tt.mod, reshaped)
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
      val upstream: Expression[V, Plus[Min[Plus[K, M], M], L], Plus[Min[Min[Plus[M, CK], M], L], M]] =
        self.grad(variable)
          .shiftLeft[M]
          .transpose[M, L]
      upstream.asInstanceOf[Expression[V, Plus[K, L], Plus[M, Min[CK, L]]]]
    }

  }

  case class ShiftRightExpression[
  V: ClassTag : Semiring,
  K <: Nat,
  CK <: Nat,
  L <: Nat : ToInt
  ](self: Expression[V, K, CK])
    extends Expression[V, Min[K, L], Plus[CK, L]] {

    val ringV = self.ringV
    val ctV = self.ctV

    override val tt = self.tt.shiftRight[L]

    override def eval() = {
      val tensor = self.eval()
      val data = tensor.data
      val reshaped = data.reshape(tt.dom.size, tt.mod.size, View.Require)
      new MyTensor[V, Min[K, L], Plus[CK, L]](tt.dom, tt.mod, reshaped)
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
      val upstream = self.grad(variable)
        .transpose[L, M]
        .shiftRight[M]
      upstream.asInstanceOf[Expression[V, Min[K, L], Plus[M, Plus[CK, L]]]]
    }
  }

  case class TransposeExpression[
  V: ClassTag : Semiring,
  K <: Nat,
  CK <: Nat,
  L <: Nat : ToInt,
  CL <: Nat : ToInt
  ](orig: Expression[V, K, CK])
    extends Expression[V, Plus[Min[K, L], CL], Plus[Min[CK, CL], L]] {

    val ringV = orig.ringV
    val ctV = orig.ctV

    override val tt = orig.tt.transpose[L, CL]

    override def eval() = {
      val tensor = orig.eval()
      val data = tensor.data

      val l = implicitly[ToInt[L]].apply()
      val cl = implicitly[ToInt[CL]].apply()
      val blockRows = orig.tt.dom.sizes.takeRight(l).product
      val blockCols = orig.tt.mod.sizes.take(cl).product
      val newData = data.reshape(tt.dom.size, tt.mod.size, View.Copy)
      for {
        row <- 0 until orig.tt.dom.sizes.dropRight(l).product
        col <- 0 until orig.tt.mod.sizes.drop(cl).product
      } {
        val view = data(
          (row * blockRows) until ((row + 1) * blockRows),
          (col * blockCols) until ((col + 1) * blockCols)
        )
        newData(
          (row * blockCols) until ((row + 1) * blockCols),
          (col * blockRows) until ((col + 1) * blockRows)
        ) := view.t
      }
      new MyTensor[V, Plus[Min[K, L], CL], Plus[Min[CK, CL], L]](tt.dom, tt.mod, newData)
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = ???
  }

  case class PlusExpression[
  V: ClassTag : Semiring,
  K <: Nat,
  CK <: Nat
  ](left: Expression[V, K, CK], right: Expression[V, K, CK])
    extends Expression[V, K, CK] {
    assert(left.tt == right.tt)

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    override val tt = left.tt

    override def eval() = {
      val leftTensor = left.eval()
      val rightTensor = right.eval()
      new MyTensor[V, K, CK](tt.dom, tt.mod, leftTensor.data +:+ rightTensor.data)
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
      val leftGrad = left.grad(variable)
      val rightGrad = right.grad(variable)
      leftGrad + rightGrad
    }
  }

  case class OuterExpression[
  V: ClassTag : Semiring,
  KL <: Nat,
  CKL <: Nat,
  KR <: Nat,
  CKR <: Nat
  ](left: Expression[V, KL, CKL], right: Expression[V, KR, CKR])
    extends Expression[V, Plus[KL, KR], Plus[CKR, CKL]] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    override val tt = left.tt.outer(right.tt)

    override def eval() = {
      val leftMatrix = left.eval().data
      val rightMatrix = right.eval().data
      val newMatrix = DenseMatrix.zeros[V](tt.dom.size, tt.mod.size)
      for {
        rowLeft <- 0 until left.tt.dom.size
        rowRight <- 0 until right.tt.dom.size
        colLeft <- 0 until left.tt.mod.size
        colRight <- 0 until right.tt.mod.size
      } {
        val entry = ringV.*(
          leftMatrix(rowLeft, colLeft),
          rightMatrix(rowRight, colRight)
        )
        newMatrix(
          rowLeft * right.tt.dom.size + rowRight,
          colLeft * right.tt.mod.size + colRight
        ) = entry
      }
      new MyTensor(tt.dom, tt.mod, newMatrix)
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
      val leftGrad = left.grad(variable)
        .asInstanceOf[Expression[V, KL, Plus[M, CKL]]]
      val rightGrad = right.grad(variable)

      implicit val ckToInt = right.tt.dom.toInt
      implicit val ckrToInt = right.tt.mod.toInt
      val lor = (leftGrad outer right)
        .shiftLeft[CKR]
        .transpose[CKR, M]
        .shiftRight[M]
      val leftRes: Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]] =
        lor.asInstanceOf[Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

      implicit val mpckr = new ToInt[Plus[M, CKR]] {
        override def apply() = implicitly[ToInt[M]].apply() + ckrToInt.apply()
      }
      val rightRes = (left outer rightGrad)
        .asInstanceOf[Expression[V, Plus[KL, KR], Plus[M, Plus[CKR, CKL]]]]

      leftRes + rightRes
    }

  }

  case class ConstantExpression[V: ClassTag : Semiring, K <: Nat, CK <: Nat]
  (
    value: MyTensor[V, K, CK]
  ) extends Expression[V, K, CK] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val tt = TensorShape(value.dom, value.mod)

    override def eval() = value

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]) = {
      new ConstantExpression[V, K, Plus[M, CK]](MyTensor[V, K, Plus[M, CK]](tt.dom, Domain.join(variable.dom, tt.mod)))
    }
  }

  case class Variable[V: Semiring : ClassTag, L <: Nat : ToInt](dom: Domain[L])
    extends Expression[V, L, Zero] {

    val ringV = implicitly[Semiring[V]]
    val ctV = implicitly[ClassTag[V]]

    val tt = TensorShape(dom, Domain())

    override def eval() = {
      throw new NotImplementedError("Variable should not be evaluated")
    }

    override def grad[M <: Nat : ToInt](variable: Variable[V, M]): Expression[V, L, Plus[M, Nat._0]] = {
      if (this eq variable) {
        new ConstantExpression[V, L, L](MyTensor.eye[V, L](dom)) // ugh, but cast will succeed
      } else {
        new ConstantExpression[V, L, M](MyTensor[V, L, M](dom, variable.dom))
      }
    }.asInstanceOf[Expression[V, L, Plus[M, Nat._0]]]
  }

}

class TensorsSpec extends FlatSpec {

  import Tensors._

  "ed" should "have a reasonable type" in {
    val ed = Domain.join(Domain(2), Domain(3))
    println(ed.sizes)

    val a = Domain[Succ[Zero]](Seq(1))
    val b = Domain[Succ[Succ[Zero]]](Seq(1))
    val c: Domain[Nat._1] = a
    assert(ed == Domain[Nat._2](sizes = List(2, 3)))
  }

  "variable" should "have covariant gradient" in {
    val a = Variable[Float, Nat._1](Domain(1))
    val b = Variable[Float, Nat._1](Domain(2))
    val t = Variable[Float, Nat._1](Domain(3))
    val c = a.grad(a)
    val d: Expression[Float, Nat._1, Nat._1] = a.grad(b)
    val x: Expression[Float, Nat._2, Nat._1] = (a outer b) grad t
  }

}
