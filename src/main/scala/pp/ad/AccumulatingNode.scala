package pp.ad

import scalaz.Scalaz
import scalaz.Scalaz.Id

case class AccumulatingNode[U[_] : ContainerOps, V, S, A](up: Node[U, V, S], rf: CollectValueFunc[V])(implicit st: ValueOps[U, V, S], val vo: ValueOps[Id, V, Any]) extends Node[Id, V, Any] {

  override implicit val ops = ContainerOps.idOps

  override val shape: Shape = null

  override implicit val vt = vo.bind(shape)

  private val opsU = implicitly[ContainerOps[U]]

  override def value = opsU.foldLeft(up())(rf.start)(rf.apply)

  // f(f(f(zero, x1), x2), x3)
  // grad_v =>
  //   (
  //    x3' +
  //    (
  //     x2' +
  //     (x1' * f'(zero, x1))
  //    ) * f'(f(zero, x1), x2) +
  //   ) * f'(f(f(zero, x1), x2), x3)

  // ug = (x1', x2', x3')

  override def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]) = {
    val opsW = implicitly[ContainerOps[W]]
    val ug = up.grad(v)
    val result = opsW.map(ug) { g =>
      val vg = opsU.zipMap(up(), g)((_, _))
      opsU.foldLeft(vg)((rf.start, vt.zero)) {
        (acc, x) =>
          val (av, ag) = acc
          val (xv, xg) = x
          (
            rf(av, xv),
            vt.times(vt.plus(xg, ag), rf.grad(av, xv))
          )
      }
    }
    opsW.map(result)(_._2)
  }
}
