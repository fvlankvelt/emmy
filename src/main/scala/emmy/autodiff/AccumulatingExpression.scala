package emmy.autodiff

import scalaz.Scalaz.Id

case class AccumulatingExpression[U[_] : ContainerOps, V, S, A](up: Expression[U, V, S], rf: CollectValueFunc[V])
                                                               (implicit fl: Floating[V], val so: ScalarOps[Double, V])
  extends Expression[Id, V, Any] {

  override val ops = ContainerOps.idOps

  override val vt = Evaluable.fromConstant(ValueOps[Id, V, Any](fl, ContainerOps.idOps, null))

  private val opsU = implicitly[ContainerOps[U]]

  override val parents = Seq(up)

  override def apply(ec: EvaluationContext) = {
    opsU.foldLeft(ec(up))(rf.start)(rf.apply)
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

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, T])(implicit  wOps: ContainerOps.Aux[W, T]) = {
    implicit val sod = so
    val ug = gc(up, v)
    val valT = vt(gc)
    val valD = valT.forDouble
    val result = wOps.map(ug) { g =>
      val vg = opsU.zipMap(gc(up), g)((_, _))
      opsU.foldLeft(vg)((rf.start, valD.zero)) {
        (acc, x) =>
          val (av, ag) = acc
          val (xv, xg) = x
          (
            rf(av, xv),
            sod.times(valD.plus(xg, ag), rf.grad(av, xv))
          )
      }
    }
    wOps.map(result)(_._2)
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
