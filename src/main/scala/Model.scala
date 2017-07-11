import breeze.linalg.Tensor

import scala.collection.mutable

sealed trait VariableType
case object Constant extends VariableType
case object Global extends VariableType
case object Local extends VariableType

case class Context
(
  constants: Map[VariableLike[_], _] = Map.empty,
  variables: mutable.Map[VariableLike[_], Any] = mutable.Map.empty
) {
  def eval[K](variable: VariableLike[K]): K = {
    constants.get(variable)
      .orElse(variables.get(variable))
      .getOrElse{
        val value = variable.eval(this)
        variables(variable) = value
        value
      }.asInstanceOf[K]
  }
}

class MeanField {

  val values: Map[VariableLike[_], (_, _)] = Map.empty
}

class Model {

  private var context: Context = Context()

  val variables: Set[VariableLike[_]] = Set.empty

  def withConstant[K](variable: VariableLike[K], value: K) : Unit = {
    context = context.copy(constants = context.constants + (variable -> value))
  }

  def fit(data: Map[VariableLike[_], _]) : MeanField = {
    new MeanField
  }
}
