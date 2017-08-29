package pp.tensor

trait Expression[K <: Nat, CK <: Nat] {

  def shape: TensorShape[K, CK]

  def eval(): Tensor[K, CK]

  def grad[M <: Nat : ToInt](variable: Variable[M]): Expression[K, Plus[M, CK]]

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
  def shiftLeft[M <: Nat : ToInt]: Expression[Plus[K, M], Min[CK, M]] = ShiftLeftExpression(this)

  def shiftRight[M <: Nat : ToInt]: Expression[Min[K, M], Plus[CK, M]] = ShiftRightExpression(this)

  def transpose[L <: Nat : ToInt, CL <: Nat : ToInt]: Expression[Plus[Min[K, L], CL], Plus[Min[CK, CL], L]] =
    TransposeExpression[K, CK, L, CL](this)

  def broadcastCov[L <: Nat : ToInt](mod: Domain[L]): Expression[K, Plus[L, CK]] =
    broadcast[Nat._0, L](Domain(), mod).asInstanceOf[Expression[K, Plus[L, CK]]]

  def broadcast[L <: Nat : ToInt, CL <: Nat : ToInt](dom: Domain[L], mod: Domain[CL]): Expression[Plus[K, L], Plus[CL, CK]] =
    outer(ConstantExpression(Tensor.ones(dom, mod)))

  def unary_- : Expression[K, CK] = ScaleExpression(this, -1)

  def +(other: Expression[K, CK]): Expression[K, CK] = PlusExpression(this, other)

  def -(other: Expression[K, CK]): Expression[K, CK] = MinExpression(this, other)

  def *(other: Expression[K, CK]): Expression[K, CK] = MulExpression(this, other)

  def *(scale: Float) = ScaleExpression(this, scale)

  def /(other: Expression[K, CK]): Expression[K, CK] = DivExpression(this, other)

  def /(scale: Float) = ScaleExpression(this, 1.0f / scale)

  def ^(other: Expression[K, CK]): Expression[K, CK] = PowExpression(this, other)

  def outer[OK <: Nat : ToInt, OCK <: Nat : ToInt](other: Expression[OK, OCK]): Expression[Plus[K, OK], Plus[OCK, CK]] =
    new OuterExpression[K, CK, OK, OCK](this, other)
}

