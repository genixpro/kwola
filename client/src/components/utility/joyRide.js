import React, { Component } from 'react';
import Joyride from 'react-joyride';
import 'react-joyride/lib/react-joyride-compiled.css';

const steps = [
  {
    title: 'Collapse',
    text: '<p>Collapse<p>',
    selector: '#topbarCollapsed',
    position: 'top',
    type: 'click',
  },
  {
    title: 'topbarAdd2Cart',
    text: '<p>topbarAdd2Cart<p>',
    selector: '#topbarAdd2CartButton',
    position: 'top',
    type: 'click',
  },
  {
    title: 'Topbar',
    text: '<p>Topbar<p>',
    selector: '#topbarUserIcon',
    position: 'top',
    type: 'click',
  },
  {
    title: 'themeSwitcher',
    text: '<p>themeSwitcher<p>',
    selector: '#themeSwitcherButton',
    position: 'top',
    type: 'click',
  },
];
export default class extends Component {
  state = {
    joyrideOverlay: true,
    joyrideType: 'continuous',
    isRunning: false,
    stepIndex: 0,
    selector: '',
  };
  callback() {}
  componentDidMount() {
    setTimeout(() => {
      this.setState({
        isRunning: true,
      });
    }, 1000);
  }
  render() {
    const { isRunning, joyrideOverlay, joyrideType, stepIndex } = this.state;

    const options = {
      callback: this.callback,
      debug: false,
      locale: {
        back: <span>Back</span>,
        close: <span>Close</span>,
        last: <span>Last</span>,
        next: <span>Next</span>,
        skip: <span>Skip</span>,
      },
      run: isRunning,
      showOverlay: joyrideOverlay,
      showSkipButton: true,
      showStepsProgress: true,
      stepIndex: stepIndex,
      steps: steps,
      type: joyrideType,
    };
    return <Joyride {...options} />;
  }
}
