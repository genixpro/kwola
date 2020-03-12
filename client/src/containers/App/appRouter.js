import React, { Component, lazy, Suspense } from 'react';
import Route from '../../components/utility/customRoute';
import Loader from '../../components/utility/Loader/';

const routes = [
  {
    path: '',
    component: lazy(() => import('../Dashboard')),
  },
  {
    path: 'new-application',
    component: lazy(() => import('../NewApplication')),
  },
  {
    path: 'applications',
    component: lazy(() => import('../ListApplications')),
  },
  {
    path: 'applications/:id',
    component: lazy(() => import('../ViewApplication')),
  },
  {
    path: 'testing_sequences/:id',
    component: lazy(() => import('../ViewTestingSequence')),
  },
  {
    path: 'training_sequences/:id',
    component: lazy(() => import('../ViewTrainingSequence')),
  },
  {
    path: 'training_steps/:id',
    component: lazy(() => import('../ViewTrainingStep')),
  },
  {
    path: 'execution_sessions/:id',
    component: lazy(() => import('../ViewExecutionSession')),
  },
  {
    path: 'execution_traces/:id',
    component: lazy(() => import('../ViewExecutionTrace')),
  }
];

class AppRouter extends Component {
  render() {
    const { url, style } = this.props;
    return (
      <Suspense fallback={<Loader />}>
        <div style={style}>
          {routes.map(singleRoute => {
            const { path, exact, ...otherProps } = singleRoute;
            return (
              <Route
                exact={exact === false ? false : true}
                key={singleRoute.path}
                path={`${url}/${singleRoute.path}`}
                {...otherProps}
              />
            );
          })}
        </div>
      </Suspense>
    );
  }
}

export default AppRouter;
