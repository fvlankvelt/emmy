package emmy.inference.aevb

import emmy.autodiff.{ CategoricalVariable, ContinuousVariable, Node, Parameter, Visitor }
import emmy.distribution.Factor

class VariableCollector(
    var visited: Set[Node]
)
  extends Visitor[VariableCollector] {

  var parameters: Set[ParameterOptimizer] = Set.empty
  var variables: Set[VariablePosterior] = Set.empty
  var factors: Seq[Factor] = Seq.empty

  var deps: Map[Node, Set[Node]] = visited.map { n ⇒
    n -> Set.empty[Node]
  }.toMap

  // stack of nodes, from the end result down
  var stack: List[Node] = List.empty

  private[aevb] def collectVars(nodes: Seq[Node]): VariableCollector = {
    nodes.foreach { p ⇒
      // add nodes up to the first variable as descendants of p
      //      val variableIndex = stack.indexWhere {
      //        case _: Factor => true
      //        case _ => false
      //      }
      //      val bottom = stack.take(variableIndex + 1)
      if (deps.contains(p)) {
        deps += (p -> (deps(p) ++ stack.headOption.toSet))
      }
      else {
        visited += p
        deps += p -> stack.headOption.toSet
        stack = p :: stack
        p.visit(this)
        stack = stack.tail
      }
    }
    this
  }

  def collect(nodes: Seq[Node]): (Set[Node], Set[VariablePosterior], Set[ParameterOptimizer], Seq[Factor], Map[Node, Set[Node]]) = {
    collectVars(nodes) // ignore result
    (visited, variables, parameters, factors, deps)
  }

  def descendants(p: Node): Set[Factor] = {
    deps(p).flatMap {
      case f: Factor ⇒ Set(f)
      case n: Node   ⇒ descendants(n)
    }
  }

  override def visitParameter[U[_], S](p: Parameter[U, S]): VariableCollector = {
    //    parameters += ParameterHolder(p)
    throw new NotImplementedError("Parameter Optimization is not implemented yet")
  }

  override def visitContinuousVariable[U[_], S](v: ContinuousVariable[U, S]): VariableCollector = {
    factors :+= v
    val posterior = ContinuousVariablePosterior(v, v)
    variables += posterior

    visitFactor(v)
  }

  override def visitCategoricalVariable(v: CategoricalVariable): VariableCollector = {
    val posterior = CategoricalVariablePosterior(v, v)
    variables += posterior

    visitFactor(v)
  }

  override def visitFactor(f: Factor): VariableCollector = {
    factors :+= f
    collectVars(f.parents)
    this
  }

  override def visitNode(n: Node): VariableCollector =
    collectVars(n.parents)

}
