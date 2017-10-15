package emmy.autodiff

import scalaz.Scalaz.Id

case class LiftedContainer[U[_], V, S](value: U[Expression[Id, V, Any]])
                                      (implicit
                                       fl: Floating[V],
                                       soo: ScalarOps[Double, V],
                                       val ops: ContainerOps.Aux[U, S]) extends Expression[U, V, S] {

  override implicit val so: ScalarOps[U[Double], U[V]] = ScalarOps.liftBoth(soo, ops)

  override implicit val vt: Evaluable[ValueOps[U, V, S]] = {
    val shape = ops.shapeOf(value)
    ValueOps(fl, ops, shape)
  }

  override def apply(ec: EvaluationContext): U[V] = {
    ops.map(value)(_ (ec))
  }

  /**
    * highly inefficient implementation of grad
    * Turns U[W[Double]] into W[U[Double]] by doing a double loop on U.
    */
  override def grad[W[_], T](gc: GradientContext, v: Variable[W, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U] = {
    val grads = ops.map(value)(gc(_, v))
    val valT = vt(gc)
    val valWT = v.vt(gc)
    val eye = ops.eye(valT.shape, 1.0, 0.0)
    val zero = wOps.fill(valWT.shape, ops.fill(valT.shape, 0.0))
    ops.foldLeft(ops.zipMap(grads, eye)((_, _)))(zero) {
      case (agg, grad) => wOps.zipMap(agg, grad._1) {
        case (wa, wb) => ops.zipMap(wa, grad._2) {
          case (a, b) => a + b * wb
        }
      }
    }
  }

  override def toString: String = {
    s"lift($value)"
  }
}

