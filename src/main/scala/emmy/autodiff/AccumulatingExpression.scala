package emmy.autodiff

import scalaz.Scalaz.Id

case class AccumulatingExpression[U[_] : ContainerOps, V, S, A](up: Expression[U, V, S], rf: CollectValueFunc[V])
                                                               (implicit vo: ValueOps[Id, V, Any])
  extends Expression[Id, V, Any] {

  implicit val st: ValueOps[U, V, S] = up.vt

  override implicit val ops = ContainerOps.idOps

  override val shape: Shape = null

  override implicit val vt = vo.bind(shape)

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

  override def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit  wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = gc(up, v)
    val result = opsW.map(ug) { g =>
      val vg = opsU.zipMap(gc(up), g)((_, _))
      opsU.foldLeft(vg)((rf.start, vt.zero)) {
        (acc, x) =>
          val (av, ag) = acc
          val (xv, xg) = x
          (
            rf(av, xv),
            vt.times(vt.plus(xg, ag), rf.grad(av, xv))
          )
      }
    }
    opsW.map(result)(_._2)
  }

  override def toString: String = {
    rf.name + "(" + up + ")"
  }
}
