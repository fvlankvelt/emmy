package pp

import breeze.linalg.DenseVector

trait PowOp[V, T <: VariableLike[V, T]] extends ((T, T) => T)

object PowOp {

  implicit object ScalarPowOp extends PowOp[Float, ScalarVariableLike] {
    def apply(upstream: ScalarVariableLike, other: ScalarVariableLike): ScalarVariableLike = {
      new ScalarVariable("**") {
        override def eval(context: Context) = {
          Math.pow(context.eval(upstream), context.eval(other)).toFloat
        }

        override def grad(scalar: ScalarVariableLike) = {
          import Function._
          val upGrad = upstream.grad(scalar).map { g =>
            val exp = other - 1.0f
            g * other * (upstream ** exp)
          }
          val otGrad = other.grad(scalar).map { g =>
            g * log(upstream) * (upstream ** other)
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          import Function._
          val upGrad = upstream.grad(vector).map { g =>
            val exp = other - 1.0f
            g * (other * (upstream ** exp)).toVector(vector.length)
          }
          val otGrad = other.grad(vector).map { g =>
            g * (log(upstream) * (upstream ** other)).toVector(vector.length)
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

  implicit object VectorPowOp extends PowOp[DenseVector[Float], VectorVariableLike] {
    def apply(upstream: VectorVariableLike, other: VectorVariableLike) = {
      new VectorVariable(upstream.length) {
        override def eval(context: Context) = {
          context.eval(upstream) ^:^ context.eval(other)
        }

        override def grad(scalar: ScalarVariableLike) = {
          import Function._
          val upGrad = upstream.grad(scalar).map { g =>
            val exp = other - VariableLike.toScalar(1.0f).toVector(g.length)
            g * other * (upstream ** exp)
          }
          val otGrad = other.grad(scalar).map { g =>
            g * log(upstream) * (upstream ** other)
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }

        override def grad(vector: VectorVariableLike) = {
          import Function._
          val upGrad = upstream.grad(vector).map { g =>
            val exp = other - VariableLike.toScalar(1.0f).toVector(g.rows)
            g * (other * (upstream ** exp)).toMatrix(vector.length)
          }
          val otGrad = other.grad(vector).map { g =>
            g * (log(upstream) * (upstream ** other)).toMatrix(vector.length)
          }
          (upGrad, otGrad) match {
            case (None, _) => otGrad
            case (_, None) => upGrad
            case _ => Some(upGrad.get + otGrad.get)
          }
        }
      }
    }
  }

}
