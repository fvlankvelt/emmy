package pp

import breeze.linalg.{DenseMatrix, DenseVector}

trait VectorVariableLike extends VariableLike[DenseVector[Float], VectorVariableLike] {

  override def repr : VectorVariableLike = this

  val length: Int

  def grad(scalar: ScalarVariableLike): Option[VectorVariableLike]

  def grad(vector: VectorVariableLike): Option[MatrixVariableLike]

  def toMatrix(cols: Int) = {
    val upstream = this
    new MatrixVariable(length, cols) {
      override def eval(context: Context) = {
        context.eval(upstream) * DenseVector.ones[Float](cols).t
      }
    }
  }

}
