package org.tensorflow.lite.examples.poseestimation.finder

import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.*

class zFinder{

    data class BodyLen(var shoulder: Float, var arr: Array<Float>, var human: Person)

    /** Pair of keypoints to draw lines between.  */
    private val bodyJoints = listOf(
        Pair(BodyPart.LEFT_ANKLE, BodyPart.LEFT_KNEE),
        Pair(BodyPart.RIGHT_ANKLE, BodyPart.RIGHT_KNEE),
        Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_HIP),
        Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_HIP),
        Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_SHOULDER),
        Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
        Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
        Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
        Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST)
    )

    fun findLengthPerson(
        person: Person
    ): Person {
        var lenBody = BodyLen(0.0f, emptyArray(), person)

        bodyJoints.forEach {
            val len: Float = sqrt(
                (person.keyPoints[it.first.position].coordinate.x - person.keyPoints[it.second.position].coordinate.x).pow(2)
                        + (person.keyPoints[it.first.position].coordinate.y - person.keyPoints[it.second.position].coordinate.y).pow(2)
            )
            lenBody.human.keyPoints[it.second.position].z = len            // for visualize
            lenBody.arr += len
        }

        lenBody.shoulder = sqrt((person.keyPoints[BodyPart.LEFT_SHOULDER.position].coordinate.x - person.keyPoints[BodyPart.RIGHT_SHOULDER.position].coordinate.x).pow(2)
                                + (person.keyPoints[BodyPart.LEFT_SHOULDER.position].coordinate.y - person.keyPoints[BodyPart.RIGHT_SHOULDER.position].coordinate.y).pow(2))

        return person
    }
}
