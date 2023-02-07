using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class IgnoreDivider : MonoBehaviour
{
    public GameObject ball;
    // Start is called before the first frame update
    void Start()
    {
        Physics.IgnoreCollision(ball.GetComponent<Collider>(), GetComponent<Collider>());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
