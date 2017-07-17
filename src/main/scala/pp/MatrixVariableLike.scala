package pp

import breeze.linalg.DenseMatrix

trait MatrixVariableLike extends VariableLike[DenseMatrix[Float], MatrixVariableLike] {
  val rows: Int
  val cols: Int

  override def repr : MatrixVariableLike = this

  private val upstream = this

  def grad(scalar: ScalarVariableLike): Option[MatrixVariableLike] = None

  def transpose: MatrixVariableLike = {
    val upstream = this
    new MatrixVariable(cols, rows) {
      override def eval(context: Context) = {
        context.eval(upstream).t
      }
    }
  }

  def mvp(vector: VectorVariableLike) = {
    assert(cols == vector.length)
    new VectorVariable(rows) {
      override def eval(context: Context) = {
        context.eval(upstream) * context.eval(vector)
      }
    }
  }
}
