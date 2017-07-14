package pp

import scala.collection.mutable


sealed trait VariableType
case object Constant extends VariableType
case object Global extends VariableType
case object Local extends VariableType

case class Assignment[K, T <: VariableLike[K, T]](key: VariableLike[K, T], value: K)

case class Context
(
  var variables: Seq[Assignment[_, _]] = Seq.empty,
  var stack: mutable.Stack[VariableLike[_, _]] = mutable.Stack.newBuilder.result()
) {
  def eval[K, T <: VariableLike[K, T]](variable: VariableLike[K, T]): K = {
    val result =
      variables
        .find(_.key eq variable)
        .map(_.value)
        .getOrElse {
          if (stack.exists(_ eq variable)) {
            throw new Exception
          }
          stack.push(variable)
          val value = variable.eval(this)
          stack.pop()
          variables :+= Assignment(variable, value)
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
