package emmy.inference.aevb

import breeze.numerics.sqrt
import emmy.autodiff._
import emmy.distribution.Factor
import emmy.inference._

case class AEVBModel(variables: Set[VariablePosterior]) extends Model {

  override def sample[U[_], V, S](n: Variable[U, V, S], ec: SampleContext): U[V] = ???

  override def update(observations: Seq[Factor]): AEVBModel = {
    // find new nodes, new variables & their (log) probability
    val collector = new VariableCollector(variables.map { v ⇒ v.O }.toSet)
    val (_, localVars, localParams, factors, globalDeps) = collector.collect(observations)

    val newGlobal = variables.map { _.next }

    val allVars = newGlobal ++ localVars

    val logpByFactor = (factors ++ allVars.map { _.P })
      .map { f ⇒ f -> f.logp }
      .toMap

    val varParams = allVars.flatMap { p ⇒
      p.parameters.map { param ⇒
        param -> Some(p)
      }
    }.toMap[ParameterOptimizer, Option[VariablePosterior]]

    val allParams = localParams.map { p ⇒
      p -> None
    }.toMap ++ varParams

    val gc = new ModelGradientContext(
      allVars.flatMap { v ⇒
        (v.O: Node, v) :: (v.P: Node, v) :: Nil
      }.toMap,
      Map.empty
    )
    val ctx = SampleContext(0, 0)

    allParams.foreach {
      case (p, Some(v)) ⇒
        val blanket = collector.descendants(v.O) + v.P
        val varLogp = blanket.map { a ⇒
          logpByFactor(a)
        }.reduceOption(_ + _)
          .getOrElse(Constant(0.0))
        val varLogq = v.logQ
        p.initialize(varLogp, varLogq, gc, ctx)

      case (p, _) ⇒
        val varLogp = (factors.map { _.logp } ++ allVars.map { _.P.logp })
          .reduceOption(_ + _)
          .getOrElse(Constant(0.0))
        val varLogq = allVars.map { _.logQ }
          .reduceOption(_ + _)
          .getOrElse(Constant(0.0))
        p.initialize(varLogp, varLogq, gc, ctx)
    }
    var iter = 1
    var delta = 0.0
    while (iter == 1 || delta > 0.001 / sqrt(observations.size)) {
      val ctx = SampleContext(iter, iter)
      delta = (for {
        param ← allParams.keys
      } yield {
        param.update(ctx)
      }).sum

      // DEBUGGING
      /*
      val params = newGlobal.toSeq
        .flatMap(_.parameters)
        .map {
          _.asInstanceOf[ParameterHolder[Id, Double]]
        }
      val mu = params(0).value.get
      val sigma = Floating.doubleFloating.exp(params(1).value.get)
      println(s"$iter $mu $sigma")
      */

      iter = iter + 1
    }

    AEVBModel(newGlobal)
  }

}

object AEVBModel {

  /**
   * Create a new model with a global set of variables to infer
   */
  def apply(global: Seq[Node]): AEVBModel = {

    // find new nodes, new variables & their (log) probability
    val collector = new VariableCollector(Set.empty)
    val (_, variables, parameters, _, _) = collector.collect(global)

    // no optimizing hyper-parameters
    assert(parameters.isEmpty)

    val logP = variables.map {
      _.P.logp
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val logQ = variables.map {
      _.logQ
    }.reduceOption(_ + _)
      .getOrElse(Constant(0.0))

    val gc = new ModelGradientContext(
      variables.map { v ⇒
        (v.P: Node, v)
      }.toMap,
      Map.empty
    )
    val ctx = SampleContext(0, 0)

    val distParams = variables.flatMap(_.parameters)
    distParams.foreach(_.initialize(logP, logQ, gc, ctx))
    var iter = 1
    var delta = 0.0
    while (iter == 1 || delta > 0.00001 || iter < 1000) {
      //    while (iter < 10000) {
      val ctx = SampleContext(iter, iter)
      delta = (for {
        param ← distParams
      } yield {
        param.update(ctx)
      }).sum

      iter = iter + 1
    }

    //    val globalSamplers = initialize(builders)
    AEVBModel(variables)
  }

}

