package pp

case class ScalarVariable(op : String) extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

case class MatrixVariable(rows: Int, cols: Int) extends MatrixVariableLike

