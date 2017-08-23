package pp.tensor

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


