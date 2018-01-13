package emmy.inference.aevb

import emmy.autodiff.{ CategoricalVariable, ContinuousVariable, Node, Parameter, Visitor }
import emmy.distribution.{ Factor, Observation }
import emmy.inference.Sampler

class VariableCollector(
    var visited: Set[Node],
    var deps:    Map[Node, Set[Node]]
)
  extends Visitor[VariableCollector] {

  var parameters: Set[ParameterOptimizer] = Set.empty
  var variables: Set[VariablePosterior] = Set.empty
  var factors: Seq[Factor] = Seq.empty

  // stack of nodes, from the end result down
  var stack: List[Node] = List.empty

  private[aevb] def collectVars(nodes: Seq[Node]): VariableCollector = {
    nodes.foreach { p ⇒
      if (visited.contains(p)) {
        //          println(": FOUND")
        val newdeps = deps.map {
          case (variable, dependants) ⇒
            if (dependants.contains(p)) {
              variable -> (dependants ++ stack.toSet)
            }
            else {
              variable -> dependants
            }
        }
        deps = newdeps
      }
      else {
        visited += p
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

  override def visitParameter[U[_], S](p: Parameter[U, S]): VariableCollector = {
    parameters += ParameterHolder(p)
    this
  }

  override def visitSampler(o: Sampler): VariableCollector = {
    factors :+= o
    collectVars(o.parents)
  }

  override def visitObservation[U[_], V, S](o: Observation[U, V, S]): VariableCollector = {
    factors :+= o
    collectVars(o.parents)
  }

  override def visitContinuousVariable[U[_], S](v: ContinuousVariable[U, S]): VariableCollector = {
    factors :+= v
    val posterior = ContinuousVariablePosterior(v, v)
    variables += posterior
    deps += v -> stack.toSet
    collectVars(v.parents)
  }

  override def visitCategoricalVariable(v: CategoricalVariable): VariableCollector = {
    factors :+= v
    val posterior = CategoricalVariablePosterior(v, v)
    variables += posterior
    deps += v -> stack.toSet
    collectVars(v.parents)
  }

  override def visitNode(n: Node): VariableCollector =
    collectVars(n.parents)

}
