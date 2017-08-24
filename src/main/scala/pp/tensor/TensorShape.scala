package pp.tensor

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

  def broadcast[L <: Nat : ToInt, CL <: Nat : ToInt](dom: Domain[L], mod: Domain[CL]): TensorShape[Plus[K, L], Plus[CL, CK]] = {
    val self = this
    implicit val varInt = new ToInt[Plus[K, L]] {
      override def apply() = self.dom.toInt.apply() + dom.toInt.apply()
    }
    implicit val covInt = new ToInt[Plus[CL, CK]] {
      override def apply() = self.mod.toInt.apply() + mod.toInt.apply()
    }
    TensorShape(
      Domain[Plus[K, L]](self.dom.sizes ++ dom.sizes),
      Domain[Plus[CL, CK]](mod.sizes ++ self.mod.sizes)
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

