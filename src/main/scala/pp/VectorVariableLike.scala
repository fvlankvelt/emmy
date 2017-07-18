package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait VectorVariableLike extends VariableLike[DenseVector[Float], VectorVariableLike] {

  override def repr : VectorVariableLike = this

  val length: Int

  def grad(scalar: ScalarVariableLike): Option[VectorVariableLike] = None

  def grad(vector: VectorVariableLike): Option[MatrixVariableLike] = {
    if (this == vector) {
      Some(new MatrixVariable(length, length) {
        override def eval(context: Context) = DenseMatrix.eye(length)
      })
    } else {
      None
    }
  }

  def toMatrix(cols: Int) = {
    val upstream = this
    new MatrixVariable(length, cols) {
      override def eval(context: Context) = {
        context.eval(upstream) * DenseVector.ones[Float](cols).t
      }
    }
  }

  def **(other: ScalarVariableLike) = {
    val upstream = this
    new VectorVariable(length) {
      override def eval(context: Context) = {
        context.eval(upstream) ^:^ context.eval(other)
      }
    }
  }

}
