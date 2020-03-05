import asyncComponent from "../helpers/AsyncFunc";

const routes = [
  {
    path: "blank-page",
    component: asyncComponent(() => import("./containers/blankPage"))
  }
];
export default routes;
