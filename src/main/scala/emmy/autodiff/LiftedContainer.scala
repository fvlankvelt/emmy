package emmy.autodiff

import scalaz.Scalaz.Id

case class LiftedContainer[U[_], V, S](value: U[Expression[Id, V, Any]])(implicit
    fl: Floating[V],
                                                                         soo:     ScalarOps[Double, V],
                                                                         val ops: ContainerOps.Aux[U, S]
) extends Expression[U, V, S] {

  override val parents = ops.foldLeft(value)(Seq.empty[Node]) { case (acc, expr) ⇒ acc :+ expr }

  override implicit val so: ScalarOps[U[Double], U[V]] = ScalarOps.liftBoth(soo, ops)

  override implicit val vt: Evaluable[ValueOps[U, V, S]] = {
    val shape = ops.shapeOf(value)
    ValueOps(fl, ops, shape)
  }

  override def eval(ec: GradientContext): Evaluable[U[V]] = {
    val evals = ops.map(value)(_.eval(ec))
    ctx => ops.map(evals)(_(ctx))
  }

  /**
   * highly inefficient implementation of grad
   * Turns U[W[Double]] into W[U[Double]] by doing a double loop on U.
   */
  override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]): Gradient[W, U] = {
    val wOps = v.ops
    val grads = ops.map(value)(gc(_, v))
    val allEmpty = ops.foldLeft(grads)(true) { case (b, g) ⇒ b && g.isEmpty }
    if (allEmpty) {
      None
    }
    else {
      Some { ctx =>
        val valT = vt(ctx)
        val valWT = v.vt(ctx)
        val eye = ops.eye(valT.shape, 1.0, 0.0)
        val zero = wOps.fill(valWT.shape, ops.fill(valT.shape, 0.0))
        ops.foldLeft(ops.zipMap(grads, eye)((_, _)))(zero) {
          case (agg, optGrad) ⇒ optGrad match {
            case (Some(grad), eyeRow) ⇒
              wOps.zipMap(agg, grad(ctx)) {
                case (wa, wb) ⇒ ops.zipMap(wa, eyeRow) {
                  case (a, b) ⇒ a + b * wb
                }
              }
            case _ ⇒ agg
          }
        }
      }
    }
  }

  override def toString: String = {
    s"lift($value)"
  }
}

