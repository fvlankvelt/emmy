package emmy.inference.aevb

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff.{ ConstantLike, ContinuousVariable, Evaluable, Expression, ScalarOps, ValueOps }

trait SamplerVariable[U[_], T]
  extends ContinuousVariable[U, T] with ConstantLike[U, Double, T] {

  override def value: Expression[U, Double, T]

  override implicit val ops: Aux[U, Shape] =
    value.ops

  override implicit val so: ScalarOps[U[Double], U[Double]] =
    value.so

  override implicit def vt: Evaluable[ValueOps[U, Double, T]] =
    value.vt
}
