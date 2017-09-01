package pp.ad

import scalaz.Scalaz
import scalaz.Scalaz.Id


trait ContainerOps[W[_]] {

  type Shape

  def shapeOf[V](value: W[V]): Shape

  def lift[A](value: A): W[A]

  def map[A, B](container: W[A])(fn: A => B): W[B]

  def zipMap[A, B, C](left: W[A], right: W[B])(fn: (A, B) => C): W[C]

  def foldLeft[A, B](container: W[A])(zero: B)(fn: (B, A) => B): B

  def eye[A](shape: Shape, one: A, zero: A): W[W[A]]

  def fill[A](shape: Shape, value: A): W[A]
}

object ContainerOps {

  type Aux[W[_], S] = ContainerOps[W] {type Shape = S}

  implicit val idOps = new ContainerOps[Id] {

    override type Shape =
      Any

    override def shapeOf[V](value: Scalaz.Id[V]) =
      null

    override def lift[A](value: A) =
      value

    override def map[A, B](container: Scalaz.Id[A])(fn: (A) => B) =
      fn(container)

    override def zipMap[A, B, C](left: Scalaz.Id[A], right: Scalaz.Id[B])(fn: (A, B) => C) =
      fn(left, right)

    override def foldLeft[A, B](container: A)(zero: B)(fn: (B, A) => B): B =
      fn(zero, container)

    override def eye[A](shape: Shape, one: A, zero: A) =
      one

    override def fill[A](shape: Shape, value: A) =
      value
  }

  implicit val listOps = new ContainerOps[List] {

    type Shape = Int

    override def shapeOf[V](value: List[V]) =
      value.size

    override def lift[A](value: A) = List(value)

    override def map[A, B](container: List[A])(fn: (A) => B) =
      container.map(fn)

    override def zipMap[A, B, C](left: List[A], right: List[B])(fn: (A, B) => C): List[C] = {
      val zipped = left.zip(right)
      zipped.map { x =>
        fn(x._1, x._2)
      }
    }

    override def foldLeft[A, B](container: List[A])(zero: B)(fn: (B, A) => B) =
      container.foldLeft(zero)(fn)

    override def eye[A](shape: Int, one: A, zero: A) = {
      Range(0, shape)
        .map { i =>
          Range(0, shape)
            .map { j =>
              if (i == j)
                one
              else
                zero
            }.toList
        }.toList
    }

    override def fill[A](shape: Int, value: A) = {
      Range(0, shape).map(_ => value).toList
    }
  }

}
