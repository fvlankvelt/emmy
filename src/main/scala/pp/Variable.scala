package pp

abstract class ScalarVariable(op : String) extends ScalarVariableLike

abstract class VectorVariable(val length: Int) extends VectorVariableLike

abstract class MatrixVariable(val rows: Int, val cols: Int) extends MatrixVariableLike

object Variable {

  def newVector(length: Int) = {
    new VectorVariable(length) {
      override def grad(scalar: ScalarVariableLike) = None

      override def grad(vector: VectorVariableLike) = None
    }
  }
}
