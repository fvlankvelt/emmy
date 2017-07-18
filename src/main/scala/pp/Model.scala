package pp

import scala.collection.mutable


sealed trait VariableType
case object Constant extends VariableType
case object Global extends VariableType
case object Local extends VariableType

case class Context
(
  var variables: Seq[(VariableLike[_, _], _)] = Seq.empty,
  var stack: mutable.Stack[VariableLike[_, _]] = mutable.Stack.newBuilder.result()
) {
  def eval[K, T <: VariableLike[K, T]](variable: VariableLike[K, T]): K = {
    val result =
      variables
        .find(_._1 eq variable)
        .map(_._2)
        .getOrElse {
          if (stack.exists(_ eq variable)) {
            throw new Exception
          }
          stack.push(variable)
          val value = variable.eval(this)
          stack.pop()
          variables :+= (variable, value)
          value
        }.asInstanceOf[K]
    result
  }
}

class MeanField {

  val values: Map[VariableLike[_, _], (_, _)] = Map.empty
}

class Model {

  var context: Context = Context()

  val variables: Set[VariableLike[_, _]] = Set.empty

  def fit(data: Iterable[Map[VariableLike[_, _], _]]) : MeanField = {
    new MeanField
  }
}
