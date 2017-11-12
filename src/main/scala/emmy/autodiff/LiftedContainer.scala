package emmy.autodiff

import scalaz.Scalaz.Id

case class LiftedContainer[U[_], V, S](value: U[Expression[Id, V, Any]])(implicit
    fl: Floating[V],
                                                                         soo:     ScalarOps[Double, V],
                                                                         val ops: ContainerOps.Aux[U, S]
) extends Expression[U, V, S] {

  override implicit val so: ScalarOps[U[Double], U[V]] = ScalarOps.liftBoth(soo, ops)

  override implicit val vt: Evaluable[ValueOps[U, V, S]] = {
    val shape = ops.shapeOf(value)
    ValueOps(fl, ops, shape)
  }

  override def apply(ec: EvaluationContext): U[V] = {
    ops.map(value)(_(ec))
  }

  /**
   * highly inefficient implementation of grad
   * Turns U[W[Double]] into W[U[Double]] by doing a double loop on U.
   */
  override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]): Option[Gradient[W, U]] = {
    val grads = ops.map(value)(gc(_, v))
    val allEmpty = ops.foldLeft(grads)(true) { case (b, g) ⇒ b && g.isEmpty }
    if (allEmpty) {
      None
    }
    else {
      val valT = vt(gc)
      val valWT = v.vt(gc)
      val eye = ops.eye(valT.shape, 1.0, 0.0)
      val zero = wOps.fill(valWT.shape, ops.fill(valT.shape, 0.0))
      Some(ops.foldLeft(ops.zipMap(grads, eye)((_, _)))(zero) {
        case (agg, optGrad) ⇒ optGrad match {
          case (Some(grad), eyeRow) ⇒
            wOps.zipMap(agg, grad) {
              case (wa, wb) ⇒ ops.zipMap(wa, eyeRow) {
                case (a, b) ⇒ a + b * wb
              }
            }
          case _ ⇒ agg
        }
      })
    }
  }

  override def toString: String = {
    s"lift($value)"
  }
}

