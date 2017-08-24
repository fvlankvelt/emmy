package pp.tensor

import breeze.math.Field

import scala.reflect.ClassTag

trait Expression[V, K <: Nat, CK <: Nat] {

  implicit val ringV: Field[V]
  implicit val ctV: ClassTag[V]

  def shape: TensorShape[K, CK]

  def eval(): Tensor[V, K, CK]

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

  def broadcastCov[L <: Nat : ToInt](mod: Domain[L]): Expression[V, K, Plus[L, CK]] =
    broadcast[Nat._0, L](Domain(), mod).asInstanceOf[Expression[V, K, Plus[L, CK]]]

  def broadcast[L <: Nat : ToInt, CL <: Nat : ToInt](dom: Domain[L], mod: Domain[CL]): Expression[V, Plus[K, L], Plus[CL, CK]] =
    outer(ConstantExpression(Tensor.ones(dom, mod)))

  def unary_- : Expression[V, K, CK] = NegExpression(this)

  def +(other: Expression[V, K, CK]): Expression[V, K, CK] = PlusExpression(this, other)

  def -(other: Expression[V, K, CK]): Expression[V, K, CK] = MinExpression(this, other)

  def *(other: Expression[V, K, CK]): Expression[V, K, CK] = MulExpression(this, other)

  def /(other: Expression[V, K, CK]): Expression[V, K, CK] = DivExpression(this, other)

  def outer[OK <: Nat : ToInt, OCK <: Nat : ToInt](other: Expression[V, OK, OCK]): Expression[V, Plus[K, OK], Plus[OCK, CK]] =
    new OuterExpression[V, K, CK, OK, OCK](this, other)
}

