package pp

import breeze.linalg.DenseMatrix

trait Distribution {
  def logp(): ScalarVariableLike
}

trait VectorDistribution extends VectorVariableLike with Distribution {

  override def grad(scalar: ScalarVariableLike): Option[VectorVariableLike] = None

  override def grad(vector: VectorVariableLike): Option[MatrixVariableLike] = {
    if (this == vector) {
      Some(new MatrixVariable(length, length) {
        override def eval(context: Context) = DenseMatrix.eye(length)
      })
    } else {
      None
    }
  }
}

trait ScalarDistribution extends ScalarVariableLike with Distribution {

  override def grad(scalar: ScalarVariableLike): Option[ScalarVariableLike] = {
    if (scalar == this) {
      Some(1.0f)
    } else {
      None
    }
  }

  override def grad(vector: VectorVariableLike): Option[VectorVariableLike] = {
    None
  }
}
