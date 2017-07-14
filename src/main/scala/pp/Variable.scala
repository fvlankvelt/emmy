package pp

import breeze.linalg._

case class ScalarVariable(op : String) extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

case class MatrixVariable(rows: Int, cols: Int) extends MatrixVariableLike

object Variable {

  implicit def toScalar(value: Float): ScalarVariable = {
    new ScalarVariable("const") {
      override def eval(context: Context) = value
    }
  }

  implicit def toVector(value: DenseVector[Float]): VectorVariable = {
    new VectorVariable(value.length) {
      override def eval(context: Context) = value
    }
  }

}
