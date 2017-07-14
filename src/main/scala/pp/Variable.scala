package pp

import breeze.linalg._

case class ScalarVariable() extends ScalarVariableLike

case class VectorVariable(length: Int) extends VectorVariableLike

case class MatrixVariable(rows: Int, cols: Int) extends MatrixVariableLike

object Variable {

  implicit def toScalar(value: Float)(implicit model: Model): ScalarVariable = {
    new ScalarVariable {
      override def eval(context: Context) = value
    }
  }

  implicit def toVector(value: DenseVector[Float])(implicit model: Model): VectorVariable = {
    new VectorVariable(value.length) {
      override def eval(context: Context) = value
    }
  }

}
