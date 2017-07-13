
sealed trait VariableType
case object Constant extends VariableType
case object Global extends VariableType
case object Local extends VariableType

case class Context
(
  var variables: Seq[(VariableLike[_], _)] = Seq.empty
) {
  def eval[K](variable: VariableLike[K]): K = {
    val result =
      variables
        .find(_._1 eq variable)
        .map(_._2)
        .getOrElse {
          val value = variable.eval(this)
          variables :+= (variable, value)
          value
        }.asInstanceOf[K]
    result
  }
}

class MeanField {

  val values: Map[VariableLike[_], (_, _)] = Map.empty
}

class Model {

  var context: Context = Context()

  val variables: Set[VariableLike[_]] = Set.empty

  def fit(data: Map[VariableLike[_], _]) : MeanField = {
    new MeanField
  }
}
